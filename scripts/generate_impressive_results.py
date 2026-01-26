"""
Deprecated entry point for curated demo visualizations.

Use scripts/demo_best_cases.py for curated best-case demos and
scripts/visualize_random_sample.py for representative random sampling.
"""

import warnings


def main() -> None:
    """Delegate to the new curated demo script with a deprecation warning."""
    warnings.warn(
        "generate_impressive_results.py is deprecated. "
        "Use scripts/demo_best_cases.py or scripts/visualize_random_sample.py instead.",
        category=DeprecationWarning,
        stacklevel=2,
    )
    from demo_best_cases import main as demo_main

    demo_main()


if __name__ == "__main__":
    main()
