#!/bin/bash

# 日本語OCRのセットアップスクリプト

echo "日本語OCRセットアップを開始します..."

# OSを検出
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    echo "macOSを検出しました。"
    
    # Homebrewがインストールされているか確認
    if ! command -v brew &> /dev/null; then
        echo "エラー: Homebrewがインストールされていません。"
        echo "https://brew.sh からHomebrewをインストールしてください。"
        exit 1
    fi
    
    # Tesseractがインストールされているか確認
    if ! command -v tesseract &> /dev/null; then
        echo "Tesseractをインストールしています..."
        brew install tesseract
    else
        echo "Tesseractは既にインストールされています。"
    fi
    
    # 日本語データをインストール
    echo "日本語OCRデータをインストールしています..."
    brew install tesseract-lang
    
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    echo "Linuxを検出しました。"
    
    # apt-getが使用可能か確認（Debian/Ubuntu）
    if command -v apt-get &> /dev/null; then
        echo "Tesseractと日本語データをインストールしています..."
        sudo apt-get update
        sudo apt-get install -y tesseract-ocr tesseract-ocr-jpn
    # yumが使用可能か確認（CentOS/RHEL）
    elif command -v yum &> /dev/null; then
        echo "Tesseractと日本語データをインストールしています..."
        sudo yum install -y tesseract tesseract-langpack-jpn
    else
        echo "エラー: サポートされていないLinuxディストリビューションです。"
        echo "手動でtesseract-ocrとtesseract-ocr-jpnをインストールしてください。"
        exit 1
    fi
    
else
    echo "エラー: サポートされていないOSです。"
    echo "手動でTesseractと日本語データをインストールしてください。"
    exit 1
fi

# インストールの確認
echo ""
echo "インストールを確認中..."
if tesseract --list-langs 2>/dev/null | grep -q "jpn"; then
    echo "✓ 日本語OCRが正常にインストールされました。"
    echo ""
    echo "使用可能な言語:"
    tesseract --list-langs
else
    echo "✗ 日本語OCRのインストールに失敗しました。"
    echo "手動でインストールしてください。"
    exit 1
fi

echo ""
echo "セットアップが完了しました！"
echo "python pdf_ocr.py <PDFファイル> を実行してOCRを使用できます。"