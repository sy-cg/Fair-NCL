"""Assemble the reviewed journal submission without experimental-data attachments."""
import hashlib
import json
import re
import shutil
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'submission_package'
PAPER = ROOT / 'els-cas-templates'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def archive(destination, paths):
    destination.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with ZipFile(destination, 'w', compression=ZIP_DEFLATED) as output:
        for path in sorted(set(paths)):
            name = path.relative_to(ROOT).as_posix()
            data = path.read_bytes()
            info = ZipInfo(name, date_time=(2026, 11, 11, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            output.writestr(info, data)
            records.append(dict(path=name, sha256=digest(data), bytes=len(data)))
    canonical = json.dumps(records, sort_keys=True, separators=(',', ':')).encode()
    return dict(archive=destination.relative_to(ROOT).as_posix(),
                content_sha256=digest(canonical), archive_sha256=digest(destination.read_bytes()),
                file_count=len(records), files=records)


def code_files():
    files = list(ROOT.glob('*.py')) + list(ROOT.glob('*.sh'))
    for directory in ('analysis', 'evaluation', 'fairness', 'methods', 'models',
                      'utils', 'tests', 'scripts', 'figures', 'tables', 'data'):
        files += list((ROOT / directory).rglob('*.py'))
    for directory in ('configs', 'experiments'):
        files += [p for p in (ROOT / directory).rglob('*')
                  if p.is_file() and p.suffix in ('.py', '.json', '.jsonl', '.sh')]
    files += list((ROOT / 'docs').glob('*.md'))
    files += [ROOT / 'docs/raw_data_sha256.txt', ROOT / 'README.md',
              ROOT / 'requirements.txt', ROOT / 'requirements-full.txt']
    files += list((ROOT / 'CoSeRec-main/src').rglob('*.py'))
    files += [ROOT / 'CoSeRec-main/LICENSE.txt', ROOT / 'CoSeRec-main/README.md']
    return [p for p in files if p.is_file() and '__pycache__' not in p.parts]


def latex_files():
    files = set()

    def visit(path):
        if path in files:
            return
        files.add(path)
        text = re.sub(r'(?<!\\)%.*', '', path.read_text(encoding='utf-8'))
        for stem in re.findall(r'\\(?:input|include)\{([^}]+)\}', text):
            visit(PAPER / (stem if stem.endswith('.tex') else stem + '.tex'))
        for stem in re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}', text):
            image = PAPER / stem
            if not image.suffix:
                image = image.with_suffix('.pdf')
            files.add(image)

    visit(PAPER / 'cas-dc-sample.tex')
    for name in ('cas-dc-marked.tex', 'build_manuscript_versions.py',
                 'README-build.md', 'revision_changes.json'):
        files.add(PAPER / name)
    for suffix in ('.bib', '.bst', '.cls', '.sty'):
        files.update(PAPER.glob('*' + suffix))
    return files


def copy(source, relative):
    destination = OUT / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def main():
    # This exact obsolete attachment is retained in the pre-edit workspace snapshot.
    obsolete_data_attachment = OUT / 'data/source_data.zip'
    if obsolete_data_attachment.is_file():
        obsolete_data_attachment.unlink()
    code = archive(OUT / 'code/fair_ncl_revision_code.zip', code_files())
    latex = archive(OUT / 'manuscript/latex_source.zip', latex_files())
    unpacked_latex = OUT / 'manuscript/latex_source'
    if unpacked_latex.is_dir():
        for record in latex['files']:
            copy(ROOT / record['path'], 'manuscript/latex_source/' + record['path'])
    copy(PAPER / 'cas-dc-sample.pdf', 'manuscript/clean.pdf')
    copy(PAPER / 'cas-dc-marked.pdf', 'manuscript/marked.pdf')
    copy(ROOT / 'point_by_point_response_filled.docx', 'response/response_to_reviewers.docx')
    copy(ROOT / 'docs/response_work/final_current/response_to_reviewers.pdf',
         'response/response_to_reviewers.pdf')
    copy(ROOT / 'Cover_Letter_Revision.docx', 'cover_letter/cover_letter.docx')
    copy(ROOT / 'Cover_Letter_Revision.pdf', 'cover_letter/cover_letter.pdf')
    copy(ROOT / 'docs/response_work/revised_locations.csv', 'revised_locations.csv')
    manuscript_hash = digest((PAPER / 'cas-dc-sample.pdf').read_bytes())
    manifest = dict(schema_version=1, release_id='fair-ncl-revision-' + code['content_sha256'][:12],
                    code_snapshot=code, latex_source=latex,
                    experimental_data_attachments='none; results are reported in the manuscript and appendices',
                    manuscript=dict(pdf='els-cas-templates/cas-dc-sample.pdf', sha256=manuscript_hash,
                                    marked_pdf='els-cas-templates/cas-dc-marked.pdf',
                                    marked_sha256=digest((PAPER / 'cas-dc-marked.pdf').read_bytes())),
                    cover_letter=dict(docx='Cover_Letter_Revision.docx',
                                      docx_sha256=digest((ROOT / 'Cover_Letter_Revision.docx').read_bytes()),
                                      pdf='Cover_Letter_Revision.pdf',
                                      pdf_sha256=digest((ROOT / 'Cover_Letter_Revision.pdf').read_bytes())),
                    public_release=dict(status='not_published_pending_author_approval',
                                        repository_url='https://github.com/sy-cg/Fair-NCL', git_commit=None),
                    historical_run_commits='not_recorded',
                    training_runs_added=0,
                    exclusion_policy='No separate experimental-result archives, raw datasets, user-level processed data, checkpoints, credentials, or review correspondence in the code candidate.')
    target = ROOT / 'docs/release/revision_submission_release.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(manifest, indent=2, ensure_ascii=False) + '\n'
    target.write_text(encoded, encoding='utf-8')
    (OUT / 'revision_submission_release.json').write_text(encoded, encoding='utf-8')
    (OUT / 'README.md').write_text(
        '# Fair-NCL revised submission\n\n'
        'This local package contains clean and yellow-marked manuscripts, the revision cover letter, the complete 17-comment response, '
        'compiled revision locations, and frozen code snapshot. '
        'It does not claim that the public repository already contains this release.\n\n'
        '- manuscript/clean.pdf: clean revised manuscript.\n'
        '- manuscript/marked.pdf: new and revised passages highlighted in yellow.\n'
        '- manuscript/latex_source.zip: complete clean/marked LaTeX sources and figure PDFs.\n'
        '- manuscript/latex_source/: when present, an identical unpacked copy of the source archive.\n'
        '- response/: clean Fair-NCL response in editable Word and PDF.\n'
        '- cover_letter/: revision cover letter in editable Word and PDF.\n'
        '- revised_locations.csv: actual manuscript labels, numbers, and printed pages.\n'
        '- code/fair_ncl_revision_code.zip: reproduction source and the licensed CoSeRec modules.\n'
        '- revision_submission_release.json: content and archive hashes; not a historical Git commit.\n\n'
        'Run sha256sum -c SHA256SUMS from this directory to verify the delivered files. '
        'The manuscript and its appendices contain the reported results; separate experimental-data files are not journal attachments. '
        'The code archive contains reproduction source, not the archived experimental records. '
        'Analysis commands requiring those records are conditional on obtaining them from the authors. '
        'Extract the code archive into a new working directory. '
        'Create docs/release/ there and copy revision_submission_release.json from this '
        'package directory to docs/release/revision_submission_release.json in the extracted '
        'root. The manifest is intentionally outside both archives to avoid self-reference. '
        'Then follow docs/REPRODUCIBILITY.md. The LaTeX source archive preserves repository-relative paths; '
        'compile from els-cas-templates/ using its README-build.md and build_manuscript_versions.py. '
        'No raw third-party data or trained checkpoints are bundled.\n\n'
        'The two manuscript versions share one annotated source. Revision locations and response page '
        'numbers refer to the clean manuscript. The marked version highlights new and revised passages '
        'in yellow; deletions are not printed. '
        'Code is the only public-release candidate in this package; the cover letter, response, and reviewer correspondence '
        'remain separate journal materials. External publication has not been performed.\n', encoding='utf-8')
    allowed = {'README.md', 'verification.json', 'revised_locations.csv',
               'revision_submission_release.json', 'code/fair_ncl_revision_code.zip',
               'manuscript/latex_source.zip', 'manuscript/clean.pdf', 'manuscript/marked.pdf',
               'response/response_to_reviewers.docx', 'response/response_to_reviewers.pdf',
               'cover_letter/cover_letter.docx', 'cover_letter/cover_letter.pdf'}
    if unpacked_latex.is_dir():
        allowed.update('manuscript/latex_source/' + record['path'] for record in latex['files'])
    delivered = [p for p in OUT.rglob('*') if p.is_file() and p.name != 'SHA256SUMS']
    unexpected = [p.relative_to(OUT).as_posix() for p in delivered
                  if p.relative_to(OUT).as_posix() not in allowed]
    if unexpected:
        raise RuntimeError('Unexpected files in submission staging: ' + ', '.join(unexpected))
    sums = ''.join(f'{digest(p.read_bytes())}  {p.relative_to(OUT).as_posix()}\n'
                   for p in sorted(delivered))
    (OUT / 'SHA256SUMS').write_text(sums, encoding='utf-8')
    for line in sums.splitlines():
        expected, name = line.split('  ', 1)
        assert digest((OUT / name).read_bytes()) == expected, name
    with ZipFile(OUT / 'code/fair_ncl_revision_code.zip') as handle:
        names = set(handle.namelist())
        required = {'run_research_pipeline.py', 'data/dataset.py', 'data/unified_preprocessing.py',
                    'CoSeRec-main/src/data_augmentation.py', 'CoSeRec-main/src/modules.py',
                    'CoSeRec-main/LICENSE.txt'}
        assert required <= names, required - names
        assert not any(name.endswith(('.docx', '.pkl', '.pt', '.dat', '.env')) for name in names)
        assert not any('response_work' in name or 'point_by_point_response' in name for name in names)
    print(json.dumps(dict(release_id=manifest['release_id'], code_files=code['file_count'],
                          data_files=0, latex_files=latex['file_count'],
                          delivered_files=len(delivered), checksums_verified=True,
                          public_release=manifest['public_release']['status']), indent=2))


if __name__ == '__main__':
    main()
