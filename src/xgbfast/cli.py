import argparse
import json
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(prog="xgbfast", description="Compile XGBoost models for direct native inference")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("compile", help="Compile and validate a JSON/UBJ model")
    build.add_argument("model")
    build.add_argument("--output", "-o", required=True)
    build.add_argument("--validate-data", help="Optional representative 2D .npy input")
    build.add_argument("--toolchain")
    build.add_argument("--jobs", type=int, default=4)
    check = commands.add_parser("validate", help="Compare compiled predictions with XGBoost")
    check.add_argument("artifact")
    check.add_argument("--data", help="Optional representative 2D .npy input")
    inspect = commands.add_parser("inspect", help="Check artifact integrity/host compatibility and display schema")
    inspect.add_argument("artifact")
    args = parser.parse_args(argv)
    try:
        if args.command == "compile":
            from . import compile_model
            output = compile_model(args.model, args.output, validation_data=args.validate_data,
                                   toolchain=args.toolchain, jobs=args.jobs)
            print(f"Ready: {output}")
        elif args.command == "validate":
            from . import validate_model
            print(json.dumps(validate_model(args.artifact, args.data), indent=2))
        else:
            from .artifact import inspect_model
            print(json.dumps(inspect_model(args.artifact), indent=2))
    except Exception as error:
        print(f"xgbfast: {error}", file=sys.stderr)
        return 1
    return 0
