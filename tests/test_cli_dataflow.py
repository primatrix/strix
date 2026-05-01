from strix.cli import build_arg_parser


def test_arg_parser_accepts_dataflow_output():
    ap = build_arg_parser()
    args = ap.parse_args(["some_file.txt", "--dataflow-output", "out.dot"])
    assert args.dataflow_output == "out.dot"


def test_arg_parser_default_is_none():
    ap = build_arg_parser()
    args = ap.parse_args(["some_file.txt"])
    assert args.dataflow_output is None
