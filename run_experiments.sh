echo === Running Fair-NCL debug experiments ===

python main_ffvae_comparison.py --model sasrec --method fair_ncl --debug
python main_ffvae_comparison.py --model gru4rec --method fair_ncl --debug

echo.
echo === All experiments completed ===
pause
