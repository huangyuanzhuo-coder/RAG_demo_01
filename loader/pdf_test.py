import time

import pdf4llm


def pdf2mk(filepath):
    start_time = time.time()
    md_text = pdf4llm.to_markdown(filepath)

    # now work with the markdown text, e.g. store as a UTF8-encoded file
    import pathlib
    output_file = filepath.split("/")[-1].replace(".PDF", ".md")
    pathlib.Path(f"./md/{output_file}").write_bytes(md_text.encode())
    end_time = time.time()

    print(end_time - start_time)


if __name__ == '__main__':
    filepath = "../bs_challenge_financial_14b_dataset/pdf/03c625c108ac0137f413dfd4136adb55c74b3805.PDF"
    # pdf2mk(filepath)

    from langchain.text_splitter import MarkdownTextSplitter

    with open("output.md", "r", encoding="utf-8") as f:
        docs = f.read()

    splitter = MarkdownTextSplitter(chunk_size=256, chunk_overlap=20)
    splits = splitter.split_text(docs)
    for i in splits[:200]:
        print(i)
        print("-" * 60)