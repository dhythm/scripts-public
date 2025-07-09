import pdfplumber

def extract_text_from_pdf(pdf_path, output_txt_path):
    with pdfplumber.open(pdf_path) as pdf:
        all_text = ""
        for page_number, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                all_text += f"--- Page {page_number + 1} ---\n"
                all_text += text + "\n\n"

    # テキストをファイルに保存
    with open(output_txt_path, mode='w', encoding='utf-8') as output_file:
        output_file.write(all_text)

    print(f"テキスト抽出完了！結果は「{output_txt_path}」に保存されました。")

# 使用例
pdf_path = "input.pdf"
output_txt_path = "output.txt"
extract_text_from_pdf(pdf_path, output_txt_path)