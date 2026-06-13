#!/usr/bin/env python3
"""
步骤2:加载"思维画像 + 全部原文",与作者的"思维分身"对话或写文章。

把画像和原文放进 system prompt 并开启 prompt caching:第一次请求会写一次
缓存(略贵),之后每次对话/写作命中缓存,价格约为正常输入价的 0.1 倍。

用法:
    python chat.py

会话内命令:
    /write <题目>   让分身就该题目写一篇完整原创文章
    /reset          清空当前对话(画像和原文仍在)
    /quit           退出
"""
import pathlib

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

MODEL = "claude-opus-4-8"
BASE = pathlib.Path(__file__).parent
ART_DIR = BASE / "articles"
PROFILE = BASE / "profile" / "thinking_profile.md"

INSTRUCTION = """\
你是下面这位思想者的"思维分身"。你已经深度内化了他的世界观、论证招式、\
心智模型和认知风格(见下面的【思维框架画像】与【原文】)。

从现在起,你不是在"模仿文风",而是"以他的方式思考":

- 面对任何问题,先用他的透镜把问题重新框定,再用他惯用的论证招式展开。
- 输出他那个层次的洞见——锋利、世故、看穿表象之下的利益与人性结构。
- 不说教、不和稀泥、不刻意两面讨好、不堆砌"正确的废话"。
- 当你拿不准时,宁可给出一个明确、有锋芒的判断,也不要退回一个安全的、\
  平均的答案。
- 写文章时给出原创的判断与论证,绝不复述下面的原文;原文只是你思维与表达\
  的根据,不是用来抄的素材。

你可以与用户正常对话,也可以在被要求时写出完整文章。\
"""


def load_articles():
    files = sorted(
        p for p in ART_DIR.glob("**/*")
        if p.suffix.lower() in (".txt", ".md") and p.name != ".gitkeep"
    )
    parts = []
    for i, p in enumerate(files, 1):
        parts.append(f"<文章{i}>\n{p.read_text(encoding='utf-8').strip()}\n</文章{i}>")
    return files, "\n\n".join(parts)


def build_system():
    if PROFILE.exists():
        profile = PROFILE.read_text(encoding="utf-8").strip()
    else:
        profile = "(尚未生成画像。建议先运行 python extract_profile.py)"
    files, corpus = load_articles()
    system = (
        INSTRUCTION
        + "\n\n# 思维框架画像\n\n" + profile
        + "\n\n# 作者的全部原文(你思维与表达的根据)\n\n" + corpus
    )
    return system, len(files)


def main():
    client = Anthropic()
    system_text, n_articles = build_system()
    # 把整块大语料标记为可缓存:命中后输入成本约 0.1 倍
    system_blocks = [{
        "type": "text",
        "text": system_text,
        "cache_control": {"type": "ephemeral"},
    }]

    history = []
    print(f"✓ 已加载 {n_articles} 篇原文 + 思维画像。作者分身就绪。")
    print("  /write <题目> 写文章   /reset 重置对话   /quit 退出\n")

    while True:
        try:
            user = input("你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user == "/quit":
            break
        if user == "/reset":
            history = []
            print("(已重置对话,画像与原文仍在)\n")
            continue
        if user.startswith("/write "):
            topic = user[len("/write "):].strip()
            user = (
                "请就以下题目,用你内化的这位作者的思想深度和表达方式,"
                "写一篇完整、原创的文章。要给出这个层次的真实洞见,不要复述原文,"
                "不要面面俱到地和稀泥。\n\n题目:" + topic
            )

        history.append({"role": "user", "content": user})
        print("\n分身 > ", end="", flush=True)
        out = ""
        with client.messages.stream(
            model=MODEL,
            max_tokens=64000,
            thinking={"type": "adaptive"},
            system=system_blocks,
            messages=history,
        ) as stream:
            for chunk in stream.text_stream:
                print(chunk, end="", flush=True)
                out += chunk
            usage = stream.get_final_message().usage
        history.append({"role": "assistant", "content": out})

        cached = usage.cache_read_input_tokens or 0
        written = usage.cache_creation_input_tokens or 0
        fresh = usage.input_tokens or 0
        print(
            f"\n\n[缓存命中 {cached} / 缓存写入 {written} / 未缓存输入 {fresh}"
            f" / 输出 {usage.output_tokens} tokens]\n"
        )


if __name__ == "__main__":
    main()
