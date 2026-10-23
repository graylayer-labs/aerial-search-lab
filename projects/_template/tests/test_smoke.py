import my_approach
from my_approach import cli


def test_the_package_imports_and_its_command_parses() -> None:
    assert my_approach.__doc__
    args = cli.build_parser().parse_args(
        ["train-my-approach", "rgb", "--percent", "10", "--fold", "220109_Baker"]
    )
    assert (args.camera, args.percent, args.seed) == ("rgb", 10, 7)
