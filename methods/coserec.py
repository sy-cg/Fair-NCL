"""CoSeRec open-source augmentation and NCE code in the shared pipeline.

The shared processed splits, SASRec backbone, and full-ranking evaluator are
kept fixed. Recommendation BCE is applied to the pipeline's prefix examples.
"""
from collections import Counter, defaultdict
import importlib.util
import hashlib
import math
from pathlib import Path
import pickle
import random

import torch
import torch.nn.functional as F

from .common import last_sequence_representation
from .wrappers import BackboneMethod

_SOURCE = Path(__file__).resolve().parents[1] / "CoSeRec-main" / "src"


def _source_class(filename, class_name):
    path = _SOURCE / filename
    if not path.is_file():
        raise FileNotFoundError(f"CoSeRec source is required: {path}")
    spec = importlib.util.spec_from_file_location(f"coserec_original_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


class ItemCFIUFSimilarity:
    """Train-only ItemCF-IUF neighbors using the authors' scoring formula."""

    def __init__(self, neighbors, items):
        self.neighbors = neighbors
        self.items = tuple(items)

    def most_similar(self, item, top_k=1, with_score=False):
        ranked = list(self.neighbors.get(int(item), ())[:top_k])
        if len(ranked) < top_k:
            fallback = [candidate for candidate in self.items if candidate != int(item)]
            ranked += [(candidate, 0.0) for candidate in random.sample(
                fallback, k=min(top_k - len(ranked), len(fallback))
            )]
        if not ranked:
            ranked = [(int(item), 0.0)]
        return ranked if with_score else [candidate for candidate, _ in ranked]


def build_itemcf_iuf_similarity(train_event_sequences, cache_dir, dataset, top_k=20):
    """Build from unique training histories only; never use validation/test events."""
    digest = hashlib.sha256()
    for user, sequence in sorted(train_event_sequences.items()):
        digest.update(str(user).encode("utf-8"))
        digest.update(b":")
        digest.update(",".join(map(str, sequence)).encode("utf-8"))
        digest.update(b";")
    cache_path = Path(cache_dir) / f"coserec_itemcf_iuf_{dataset}_k{top_k}_{digest.hexdigest()[:12]}.pkl"
    if cache_path.exists():
        with cache_path.open("rb") as handle:
            payload = pickle.load(handle)
        return ItemCFIUFSimilarity(payload["neighbors"], payload["items"])

    counts = Counter()
    cooccurrence = defaultdict(lambda: defaultdict(float))
    for sequence in train_event_sequences.values():
        items = sorted({int(item) for item in sequence if int(item) > 0})
        if not items:
            continue
        counts.update(items)
        weight = 1.0 / math.log1p(len(items)) if len(items) > 1 else 0.0
        for item in items:
            for other in items:
                if item != other:
                    cooccurrence[item][other] += weight

    neighbors = {}
    for item, related in cooccurrence.items():
        ranked = sorted(
            ((other, score / math.sqrt(counts[item] * counts[other]))
             for other, score in related.items()),
            key=lambda pair: (-pair[1], pair[0]),
        )
        neighbors[item] = ranked[:top_k]
    payload = {"neighbors": neighbors, "items": sorted(counts)}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return ItemCFIUFSimilarity(**payload)


class CoSeRecMethod(BackboneMethod):
    """Authors' Random augmentations and NCELoss on the shared SASRec backbone."""

    def __init__(self, backbone, config, similarity):
        super().__init__(backbone, config, name="coserec")
        RandomAugmentation = _source_class("data_augmentation.py", "Random")
        NCELoss = _source_class("modules.py", "NCELoss")
        self.augmenter = RandomAugmentation(
            tao=getattr(config, "coserec_crop_rate", 0.2),
            gamma=getattr(config, "coserec_mask_rate", 0.7),
            beta=getattr(config, "coserec_reorder_rate", 0.2),
            item_similarity_model=similarity,
            insert_rate=getattr(config, "coserec_insert_rate", 0.4),
            max_insert_num_per_pos=getattr(config, "coserec_max_insert_num_per_pos", 1),
            substitute_rate=getattr(config, "coserec_substitute_rate", 0.1),
            augment_threshold=getattr(config, "coserec_short_threshold", 4),
            augment_type_for_short="SIM",
        )
        self.nce = NCELoss(getattr(config, "coserec_temperature", 1.0), config.device)
        self.cf_weight = float(getattr(config, "coserec_cf_weight", 0.1))

    def _augment_batch(self, input_seq):
        width = input_seq.shape[1]
        augmented = []
        for row in input_seq.detach().cpu().tolist():
            sequence = [int(item) for item in row if item > 0]
            changed = self.augmenter(sequence) if sequence else sequence
            changed = changed[-width:]
            augmented.append([0] * (width - len(changed)) + changed)
        return torch.tensor(augmented, dtype=input_seq.dtype, device=input_seq.device)

    def compute_loss(self, batch):
        input_seq = batch["input_seq"]
        negatives = batch["negative_items"]
        if negatives.shape[1] < 1:
            raise ValueError("CoSeRec needs train_num_negative >= 1")
        representation = last_sequence_representation(self.backbone, input_seq)
        positive = self.backbone.item_emb(batch["target"])
        negative = self.backbone.item_emb(negatives[:, 0])
        rec_loss = (
            F.softplus(-(representation * positive).sum(-1))
            + F.softplus((representation * negative).sum(-1))
        ).mean()
        first = self.backbone(self._augment_batch(input_seq)).reshape(input_seq.shape[0], -1)
        second = self.backbone(self._augment_batch(input_seq)).reshape(input_seq.shape[0], -1)
        cl_loss = self.nce(first.float(), second.float())
        return {"loss": rec_loss + self.cf_weight * cl_loss,
                "rec_loss": rec_loss, "cl_loss": cl_loss}


