"""
Convenience entrypoint so you don't have to remember the `-m pipeline.ev_engine`
module syntax. Run from the repo root:

    python run_projections.py --season 2026
    python run_projections.py --season 2026 --push-firestore
    python run_projections.py --season 2026 --push-firestore --contenders-only

See pipeline/ev_engine.py for what each flag does and how bracket detection works.
"""

from pipeline.ev_engine import main

if __name__ == "__main__":
    main()
