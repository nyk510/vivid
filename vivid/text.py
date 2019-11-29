# encoding: utf8
"""
simple text tools
"""

from __future__ import unicode_literals

import re
import string
import unicodedata


def unicode_normalize(cls, doc):
    pt = re.compile('([{}]+)'.format(cls))

    def norm(codec):
        return unicodedata.normalize('NFKC', codec) if pt.match(codec) else codec

    doc = ''.join(norm(x) for x in re.split(pt, doc))
    doc = re.sub('－', '-', doc)
    return doc


def remove_extra_spaces(doc):
    """
    余分な空白を削除

    Args:
        doc (String)
    Return
        空白除去された文章 (String)
    """

    doc = re.sub('[ 　]+', ' ', doc)
    blocks = ''.join((
        '\u4E00-\u9FFF',  # CJK UNIFIED IDEOGRAPHS
        '\u3040-\u309F',  # HIRAGANA
        '\u30A0-\u30FF',  # KATAKANA
        '\u3000-\u303F',  # CJK SYMBOLS AND PUNCTUATION
        '\uFF00-\uFFEF'  # HALFWIDTH AND FULLWIDTH FORMS
    ))
    basic_latin = '\u0000-\u007F'

    def remove_space_between(cls1, cls2, doc):
        pt = re.compile('([{}]) ([{}])'.format(cls1, cls2))
        while pt.search(doc):
            doc = pt.sub(r'\1\2', doc)
        return doc

    doc = remove_space_between(blocks, blocks, doc)
    doc = remove_space_between(blocks, basic_latin, doc)
    doc = remove_space_between(basic_latin, blocks, doc)
    return doc


def normalize_neologd(doc):
    """
    以下の文章の正規化を行います.
        * 空白の削除
        * 文字コードの変換(utf-8へ)
        * ハイフン,波線（チルダ)の統一
        * 全角記号の半角への変換   (？→?など)

    Args:
        doc(str):
            正規化を行いたい文章

    Return(str):
        正規化された文章
    """

    doc = doc.strip()
    doc = unicode_normalize('０-９Ａ-Ｚａ-ｚ｡-ﾟ', doc)

    def maketrans(f, t):
        return {ord(x): ord(y) for x, y in zip(f, t)}

    doc = re.sub('[˗֊‐‑‒–⁃⁻₋−]+', '-', doc)  # normalize hyphens
    doc = re.sub('[﹣－ｰ—―─━ー]+', 'ー', doc)  # normalize choonpus
    doc = re.sub('[~∼∾〜〰～]', '', doc)  # remove tildes
    doc = doc.translate(
        maketrans('!"#$%&\'()*+,-./:;<=>?@[¥]^_`{|}~｡､･「」『』',
                  '！”＃＄％＆’（）＊＋，－．／：；＜＝＞？＠［￥］＾＿｀｛｜｝〜。、・｢｣｢｣'))

    doc = remove_extra_spaces(doc)
    doc = unicode_normalize('！”＃＄％＆’（）＊＋，－．／：；＜＞？＠［￥］＾＿｀｛｜｝〜', doc)  # keep ＝,・,「,」
    doc = re.sub('[’]', '\'', doc)
    doc = re.sub('[”]', '"', doc)
    doc = re.sub('[“]', '"', doc)
    return doc


class RemoveSign:
    def __init__(self):
        self.string_norm_table = str.maketrans("", "", string.punctuation + "「」、。・")

    def __call__(self, text):
        """

        Args:
            text(str): input text

        Returns:
            normalized text
        """
        text = unicodedata.normalize('NFKC', text)
        text = text.translate(self.string_norm_table)
        return text


remove_sign = RemoveSign()
