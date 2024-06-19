help () {
    echo "Usage: sh shell_scripts/run_app.sh <mode>. mode: single or multi"
}

if [ $# -ne 1 ]; then
    help
elif [ "$1" = "single" ]; then
    python app/hydit_app.py --infer-mode fa
elif [ "$1" = "multi" ]; then
    python app/multiTurnT2I_app.py
else
    help
fi
