"""agent.cfg parsing (core/common/config.py): comments after values."""

from agenticwf.agent.core.common.config import new_parser


def test_comment_after_value_is_stripped(tmp_path):
    path = tmp_path / "agent.cfg"
    path.write_text("[llm.model.qwen]\n"
                    "provider = vllm                 # OpenAI-compatible\n"
                    "base_url = https://example.org/v1#frag\n"
                    'extra = {"reasoning_effort": "high"}  # JSON\n'
                    "# a whole-line comment\n"
                    "label = Qwen\n")
    parser = new_parser()
    parser.read(path)
    section = parser["llm.model.qwen"]
    assert section["provider"] == "vllm"
    assert section["base_url"] == "https://example.org/v1#frag"   # no whitespace before '#': kept
    assert section["extra"] == '{"reasoning_effort": "high"}'
    assert section["label"] == "Qwen"
