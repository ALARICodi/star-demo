#!/usr/bin/env python3
"""
步骤1:从 articles/ 里的文章,提炼出作者的"思维框架画像"。

它不总结文章写了什么,而是还原"写这些文章的是一个怎样思考的人"——
作者的世界观、论证招式、心智模型、认知风格,使得另一个 AI 拿着这份画像
就能像他一样去思考新问题。

用法:
    python extract_profile.py

输出:
    profile/thinking_profile.md  (生成后强烈建议你人工通读一遍、删改、加料)
"""
import sys
import pathlib

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

MODEL = "claude-opus-4-8"
BASE = pathlib.Path(__file__).parent
ART_DIR = BASE / "articles"
OUT = BASE / "profile" / "thinking_profile.md"

EXTRACT_SYSTEM = """\
你是一位极其敏锐、世故、不讲废话的思想分析师。下面会给你同一位作者的多篇\
文章(议论或随笔类思想性文字,题材多为政商关系、人情世故、社会洞察)。

你的任务【不是】总结这些文章写了什么,而是还原"写这些文章的是一个怎样思考\
的人"——提炼出他可迁移的思维框架,使得另一个人拿着你的提炼,就能像他一样\
去思考一个全新的问题。

请刻意忽略具体话题,聚焦在"思维方式"本身。输出一份结构化画像,包含:

1. **核心世界观与价值预设**:他默认相信什么?对人性、权力、利益、关系、\
   规则的底层假设是什么?他眼里世界是怎么运转的?

2. **惯用的论证招式**:他如何展开一个观点?(例如:先抛一个反常识判断、\
   从一个具体场景切入、揭开表象之下的利益结构、用对比或反讽收口……)\
   尽量具体,并引用他文中的真实例子佐证。

3. **心智模型 / 分析透镜**:他习惯用哪些框架看问题?(如博弈、人情账、\
   位置决定立场、长期 vs 短期、明规则与潜规则……)

4. **如何重新框定问题**:面对一个常见现象,他通常从哪个别人想不到的角度\
   切入?他的"刁钻视角"有什么规律?

5. **认知风格与态度**:锋利 / 冷峻 / 世故 / 不站队?他刻意避免什么(说教、\
   和稀泥、两面讨好、正确的废话)?

6. **表达特征**:句式、节奏、用词的标志性习惯。(这是次要项,服务于上面的\
   思想,不要喧宾夺主。)

要求:具体、可操作、带原文佐证。把它写成一份"如何成为这个思考者"的操作\
手册,而不是一篇赏析。直接输出画像正文,不要客套开场白。\
"""


def load_articles():
    files = sorted(
        p for p in ART_DIR.glob("**/*")
        if p.suffix.lower() in (".txt", ".md") and p.name != ".gitkeep"
    )
    if not files:
        sys.exit(f"× 没有在 {ART_DIR} 找到文章。先把 .txt / .md 文章丢进去(一篇一个文件)。")
    parts = []
    for i, p in enumerate(files, 1):
        text = p.read_text(encoding="utf-8").strip()
        parts.append(f'<文章{i} 标题="{p.stem}">\n{text}\n</文章{i}>')
    return files, "\n\n".join(parts)


def main():
    files, corpus = load_articles()
    client = Anthropic()
    print(f"✓ 读到 {len(files)} 篇文章,正在用 {MODEL} 提炼思维内核……\n")
    print("=" * 60)

    out_text = ""
    with client.messages.stream(
        model=MODEL,
        max_tokens=16000,
        thinking={"type": "adaptive"},
        system=EXTRACT_SYSTEM,
        messages=[{"role": "user", "content": corpus}],
    ) as stream:
        for chunk in stream.text_stream:
            print(chunk, end="", flush=True)
            out_text += chunk
        usage = stream.get_final_message().usage

    print("\n" + "=" * 60)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(out_text, encoding="utf-8")
    print(f"✓ 画像已保存到 {OUT}")
    print(f"  (输入 {usage.input_tokens} tokens / 输出 {usage.output_tokens} tokens)")
    print("→ 强烈建议你打开它通读一遍,删掉不准的、补上你更懂的,再进入步骤2。")


if __name__ == "__main__":
    main()
