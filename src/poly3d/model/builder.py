"""
モデル構築ファクトリ

train.py と sample.py の両方から使用される共通のモデル構築関数。
argparse.Namespace または dict からモデルをインスタンス化する。
"""
from __future__ import annotations

import argparse

from poly3d.model.cond_encoder import ConditionalEncoder
from poly3d.model.vae import StructuralVAE
from poly3d.model.dit import LatentDiT


def build_cond_encoder(args: argparse.Namespace) -> ConditionalEncoder:
    return ConditionalEncoder(
        hidden_dim=args.hidden_dim,
        edge_dim=args.edge_dim,
        n_layers=args.cond_layers,
        atom_emb_dim=args.atom_emb_dim,
        hyb_emb_dim=args.hyb_emb_dim,
        bond_emb_dim=args.bond_emb_dim,
        use_rwpe=args.use_rwpe,
        use_lappe=args.use_lappe,
    )


def build_vae(args: argparse.Namespace) -> StructuralVAE:
    return StructuralVAE(
        cond_dim=args.hidden_dim,
        edge_dim=args.edge_dim,
        hidden_dim=args.vae_hidden_dim,
        latent_dim=args.latent_dim,
        enc_layers=args.enc_layers,
        dec_layers=args.dec_layers,
        egt_every=getattr(args, 'egt_every', 0),
        enc_egt_every=getattr(args, 'enc_egt_every', 0),
    )


def build_dit(args: argparse.Namespace) -> LatentDiT:
    return LatentDiT(
        latent_dim=args.latent_dim,
        cond_dim=args.hidden_dim,
        time_dim=args.time_dim,
        hidden_dim=args.dit_hidden_dim,
        n_heads=args.dit_n_heads,
        n_layers=args.dit_n_layers,
        use_pos_bias=args.use_pos_bias,
    )
