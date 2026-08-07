from scripts.run_brain import build_parser, service_config_from_args


def test_brain_cli_default_task_timeout_wires_into_service_config():
    parser = build_parser()

    default_args = parser.parse_args([])
    default_config = service_config_from_args(default_args, "")
    assert default_args.default_task_timeout_ms == 270_000
    assert default_config.default_task_timeout_ms == 270_000

    demo_args = parser.parse_args(["--default-task-timeout-ms", "75000"])
    demo_config = service_config_from_args(demo_args, "")
    assert demo_config.default_task_timeout_ms == 75_000


def test_brain_cli_pipeline_generation_limit_wires_into_service_config():
    parser = build_parser()
    args = parser.parse_args(["--pipeline-max-new-tokens", "12"])

    config = service_config_from_args(args, "")

    assert config.pipeline_max_new_tokens == 12
