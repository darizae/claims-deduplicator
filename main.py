from my_timer import Timer

from claims_cleaner.cli import main

if __name__ == "__main__":
    timer = Timer()
    timer.start()
    main()
    timer.stop()
    timer.print_elapsed_time()
