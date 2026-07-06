"""
.aselmdb ファイルの直接読み込み（fairchem不要）。

OPoly26 の .aselmdb は zlib 圧縮された JSON を lmdb に格納した形式。
各エントリの配列フィールドは ASE の __ndarray__ エンコーディングを使用。
"""
from __future__ import annotations

import json
import zlib
from pathlib import Path
from typing import Generator, List, Tuple

import lmdb
import numpy as np


def _decompress(val: bytes) -> bytes:
    """zlib 圧縮されていれば展開する。

    zlib のヘッダは圧縮レベル・辞書指定によって先頭2バイトが変動する
    （例: 0x78 0x5e など）ため、特定ヘッダの列挙による判定はやめ、
    まず zlib.decompress を試して失敗したら非圧縮データとして扱う。
    """
    try:
        return zlib.decompress(val)
    except zlib.error:
        return val


def _decode_ndarray(obj) -> np.ndarray | object:
    """ASE の __ndarray__ エンコーディングをデコードする。"""
    if isinstance(obj, dict) and '__ndarray__' in obj:
        shape, dtype, flat = obj['__ndarray__']
        return np.array(flat, dtype=dtype).reshape(shape)
    return obj


def count_entries(path: str | Path) -> int:
    """lmdb ファイルのエントリ数を返す。"""
    env = lmdb.open(
        str(path), subdir=False, readonly=True,
        lock=False, readahead=False, meminit=False,
    )
    with env.begin() as txn:
        n = txn.stat()['entries']
    env.close()
    return n


def iter_lmdb(path: str | Path, stats: dict | None = None) -> Generator[dict, None, None]:
    """lmdb ファイルから生 JSON dict をひとつずつ yield する。

    Parameters
    ----------
    stats : dict, optional
        与えられた場合、無言スキップの件数をカウントして書き込む。
        - 'decode_error': JSON デコード失敗件数
        - 'missing_key' : dict でない/必須キー欠落件数
        呼び出し元はこの辞書を最終ログの集計に利用できる。
    """
    env = lmdb.open(
        str(path), subdir=False, readonly=True,
        lock=False, readahead=False, meminit=False,
    )
    with env.begin() as txn:
        cursor = txn.cursor()
        for key, val in cursor.iternext():
            raw = _decompress(val)
            try:
                obj = json.loads(raw)
            except Exception:
                if stats is not None:
                    stats['decode_error'] = stats.get('decode_error', 0) + 1
                continue

            if not isinstance(obj, dict):
                if stats is not None:
                    stats['missing_key'] = stats.get('missing_key', 0) + 1
                continue
            if 'numbers' not in obj or 'positions' not in obj or 'data' not in obj:
                if stats is not None:
                    stats['missing_key'] = stats.get('missing_key', 0) + 1
                continue

            yield obj
    env.close()


def read_molecule(
    record: dict,
) -> Tuple[np.ndarray, np.ndarray, int, str]:
    """
    JSON レコードから分子情報を取り出す。

    Returns
    -------
    atomic_nums : np.ndarray, shape (N,), dtype int
    positions   : np.ndarray, shape (N, 3), dtype float64
    charge      : int
    sid         : str
    """
    atomic_nums = _decode_ndarray(record['numbers']).flatten().astype(int)
    positions = np.asarray(_decode_ndarray(record['positions']), dtype=np.float64).reshape(-1, 3)
    if positions.shape[0] != atomic_nums.shape[0]:
        raise ValueError(
            f'原子数とpositions行数が不一致: numbers={atomic_nums.shape[0]}, '
            f'positions={positions.shape[0]}'
        )
    charge = int(record['data'].get('charge', 0))
    sid = str(record['data'].get('sid', ''))
    return atomic_nums, positions, charge, sid


def list_lmdb_files(data_dir: str | Path) -> List[Path]:
    """ディレクトリ以下の .aselmdb ファイルを列挙する（ソート済み）。"""
    data_dir = Path(data_dir)
    return sorted(data_dir.glob('*.aselmdb'))
