#!/usr/bin/env python3
"""
步骤 0.5(可选):把"抖音截图"批量 OCR 成文章文本,写进 articles/。

适合:博主内容是图文 / 视频里有大段文字,你手动截了图,想自动转成纯文本语料。
这一步同样是隔离的:图片只在本脚本里被读取,绝不进入你的主对话。

怎么放图(二选一):
  A) 一篇文章 = 一个子文件夹:
       screenshots/博主A-论政商关系/01.png  02.png  03.png ...
     脚本会按文件名顺序把同一子文件夹的多张图拼成一篇,输出 articles/博主A-论政商关系.txt
  B) 一张图 = 一篇:直接把单张图放在 screenshots/ 根目录,各自成篇。

用法:
    python ocr_to_articles.py
"""
import base64
import pathlib
import sys

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

MODEL = "claude-opus-4-8"
BASE = pathlib.Path(__file__).parent
SHOTS = BASE / "screenshots"
ART_DIR = BASE / "articles"

MEDIA = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}

OCR_PROMPT = (
    "下面是同一篇内容的一张或多张截图(来自抖音图文/视频)。请按阅读顺序,"
    "把其中的【正文文字】完整、逐字提取出来,输出纯文本。\n"
    "要求:只要正文本身;不要加任何解说、标题标注、画面描述;"
    "不要输出点赞/评论/界面按钮等无关文字;多张图是同一篇的连续内容,顺次接续。"
)


def img_block(path: pathlib.Path):
    data = base64.standard_b64encode(path.read_bytes()).decode()
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": MEDIA[path.suffix.lower()], "data": data},
    }


def collect_jobs():
    """返回 [(输出文件名stem, [图片路径,...]), ...]"""
    if not SHOTS.exists():
        sys.exit(f"× 没有 {SHOTS} 文件夹。先建好并把截图放进去。")
    jobs = []
    # A) 每个子文件夹 = 一篇
    for sub in sorted(p for p in SHOTS.iterdir() if p.is_dir()):
        imgs = sorted(p for p in sub.iterdir() if p.suffix.lower() in MEDIA)
        if imgs:
            jobs.append((sub.name, imgs))
    # B) 根目录下的散图 = 各自一篇
    for p in sorted(SHOTS.iterdir()):
        if p.is_file() and p.suffix.lower() in MEDIA:
            jobs.append((p.stem, [p]))
    if not jobs:
        sys.exit(f"× {SHOTS} 里没有图片(支持 png/jpg/jpeg/webp)。")
    return jobs


def main():
    jobs = collect_jobs()
    client = Anthropic()
    ART_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ 共 {len(jobs)} 篇待提取,用 {MODEL} OCR 中……\n")

    for idx, (stem, imgs) in enumerate(jobs, 1):
        out_path = ART_DIR / f"{stem}.txt"
        print(f"[{idx}/{len(jobs)}] {stem}  ({len(imgs)} 张图) ", end="", flush=True)
        content = [img_block(p) for p in imgs] + [{"type": "text", "text": OCR_PROMPT}]
        text = ""
        with client.messages.stream(
            model=MODEL,
            max_tokens=16000,
            messages=[{"role": "user", "content": content}],
        ) as stream:
            for chunk in stream.text_stream:
                text += chunk
        out_path.write_text(text.strip() + "\n", encoding="utf-8")
        print(f"→ 已写入 {out_path.name}  ({len(text)} 字)")

    print("\n✓ 全部完成。建议抽查 articles/ 里几篇,确认 OCR 没串行/漏字,再进步骤1。")


if __name__ == "__main__":
    main()
