import subprocess

from my_timer import Timer


def main():
    timer = Timer()
    timer.start()

    cmd = [
        "python", "-m", "claim_deduplicator.cli",
        "--input-json", "data/example_dataset.json",
        "--output-json", "data/example_dataset_deduped.json",
        "--field-to-deduplicate", "reference_acus",
        "--measure-redundancy"
    ]

    subprocess.run(cmd, check=True)

    timer.stop()
    timer.print_elapsed_time()


if __name__ == "__main__":
    main()
