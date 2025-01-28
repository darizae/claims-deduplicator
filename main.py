import subprocess

from my_timer import Timer


def main():
    timer = Timer()
    timer.start()

    cmd = [
        "python", "-m", "claims_cleaner.cli",
        "--input-json", "data/rose_datasets_small.json",
        "--output-json", "data/rose_datasets_small_deduped.json",
        "--field-to-deduplicate", "reference_acus",
        "--clusters-output", "data/clusters_analysis.json",
        "--measure-redundancy"
    ]

    subprocess.run(cmd, check=True)

    timer.stop()
    timer.print_elapsed_time()


if __name__ == "__main__":
    main()
