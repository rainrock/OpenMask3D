import argparse

def main(args):
    # Your code logic goes here
    print(f"Input file: {args.input}")
    if args.verbose:
        print("Verbose mode enabled")

if __name__ == "__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Your program description")

    # Add arguments to the parser
    parser.add_argument("input", help="Input file path")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose mode")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the main function
    main(args)

