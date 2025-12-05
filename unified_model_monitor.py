#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
통합 모델 모니터링 시스템
- 모든 실험에 대한 데이터 분석 및 시각화를 하나의 스크립트로 처리
- CSV 파일 생성 및 모든 그래프 자동 생성
- /model_result_monitoring/실험_모델_id 경로에 결과 저장
"""

import os
import sys
import pandas as pd #type: ignore
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns #type: ignore
from pathlib import Path
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ModelMonitor:
    """통합 모델 모니터링 클래스"""
    
    def __init__(self, workspace_dir, output_base_dir="model_result_monitoring"):
        self.workspace_dir = Path(workspace_dir)
        self.graph_dir = self.workspace_dir / "graph"
        self.output_base_dir = self.workspace_dir / output_base_dir
        self.output_base_dir.mkdir(exist_ok=True)
        
        # 그래프 스타일 설정
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def discover_experiments(self):
        """모든 실험 데이터를 발견하고 정리"""
        experiments = []
        
        if not self.graph_dir.exists():
            print(f"❌ 그래프 디렉토리가 없습니다: {self.graph_dir}")
            return experiments
        
        # 새 형식: graph/<실험ID>/ 형태 탐색
        for exp_dir in self.graph_dir.iterdir():
            if exp_dir.is_dir():
                epoch_file = exp_dir / "epoch-loss.txt"
                batch_file = exp_dir / "batch-loss.txt"
                
                if epoch_file.exists() and batch_file.exists():
                    experiments.append({
                        'id': exp_dir.name,
                        'path': exp_dir,
                        'epoch_file': epoch_file,
                        'batch_file': batch_file,
                        'format': 'directory'
                    })
        
        # 구 형식: graph/epoch-loss-<ID>.txt 형태 탐색 (폴백)
        if not experiments:
            epoch_files = list(self.graph_dir.glob("epoch-loss-*.txt"))
            for epoch_file in epoch_files:
                exp_id = epoch_file.stem.replace("epoch-loss-", "")
                batch_file = self.graph_dir / f"batch-loss-{exp_id}.txt"
                
                if batch_file.exists():
                    experiments.append({
                        'id': exp_id,
                        'path': self.graph_dir,
                        'epoch_file': epoch_file,
                        'batch_file': batch_file,
                        'format': 'flat'
                    })
        
        # 수정 시간 기준으로 정렬 (최신 순)
        experiments.sort(key=lambda x: x['epoch_file'].stat().st_mtime, reverse=True)
        
        print(f"🔍 발견된 실험: {len(experiments)}개")
        for exp in experiments:
            print(f"   - {exp['id']} ({exp['format']} format)")
        
        return experiments
    
    def load_experiment_data(self, experiment):
        """실험 데이터 로드"""
        try:
            # New format with header and comma separator
            if experiment['format'] == 'directory':
                epoch_df = pd.read_csv(experiment['epoch_file']) # sep=',' is default
                batch_df = pd.read_csv(experiment['batch_file'])
                # Ensure timestamp columns are parsed as strings (not datetime for display purposes)
                if 'timestamp' in epoch_df.columns:
                    epoch_df['timestamp'] = epoch_df['timestamp'].astype(str)
                if 'timestamp' in batch_df.columns:
                    batch_df['timestamp'] = batch_df['timestamp'].astype(str)
                # Convert empty val_loss to NaN for visualization
                if 'val_loss' in epoch_df.columns:
                    epoch_df['val_loss'] = pd.to_numeric(epoch_df['val_loss'], errors='coerce')
            # Old format with space separator and no header
            else: # format == 'flat'
                epoch_df = pd.read_csv(experiment['epoch_file'], sep=' ', header=None,
                                     names=['epoch', 'avg_loss'])
                batch_df = pd.read_csv(experiment['batch_file'], sep=' ', header=None,
                                     names=['epoch', 'batch_num', 'loss'])

            return epoch_df, batch_df
        
        except Exception as e:
            print(f"❌ {experiment['id']} 데이터 로드 실패: {e}")
            return None, None
    
    def calculate_epoch_statistics(self, batch_df):
        """에폭별 배치 손실 통계 계산"""
        epoch_stats = batch_df.groupby('epoch')['loss'].agg([
            'count',      # 배치 수
            'mean',       # 평균
            'std',        # 표준편차
            'var',        # 분산
            'min',        # 최솟값
            'max',        # 최댓값
            'median'      # 중앙값
        ]).reset_index()
        
        # 추가 통계 (0으로 나누기 방지)
        epoch_stats['cv'] = epoch_stats.apply(
            lambda row: row['std'] / row['mean'] if row['mean'] > 0 and row['std'] > 0 else 0, axis=1
        )
        epoch_stats['range'] = epoch_stats['max'] - epoch_stats['min']  # 범위
        
        # NaN을 0으로 대체
        epoch_stats.fillna(0, inplace=True)
        
        return epoch_stats
    
    def calculate_gradient_analysis(self, epoch_df):
        """기울기 소실 분석"""
        # Handle both old and new column names
        loss_col = 'avg_loss' if 'avg_loss' in epoch_df.columns else 'loss'
        epoch_col = 'epoch'
        
        epochs = epoch_df[epoch_col].values
        losses = epoch_df[loss_col].values
        
        # 1차, 2차 미분 계산
        dloss_depoch = np.gradient(losses, epochs)
        d2loss_depoch2 = np.gradient(dloss_depoch, epochs)
        
        # 기울기 소실 분석
        abs_gradient = np.abs(dloss_depoch)
        final_gradient_magnitude = np.mean(abs_gradient[-50:]) if len(abs_gradient) >= 50 else np.mean(abs_gradient[-10:])
        initial_gradient_magnitude = np.mean(abs_gradient[:50]) if len(abs_gradient) >= 50 else np.mean(abs_gradient[:10])
        
        gradient_ratio = final_gradient_magnitude / initial_gradient_magnitude if initial_gradient_magnitude > 0 else 0
        
        # 수렴 상태 판정
        if gradient_ratio < 0.01:
            convergence_status = "심각한 기울기 소실"
        elif gradient_ratio < 0.1:
            convergence_status = "기울기 소실 가능성"
        elif gradient_ratio < 0.5:
            convergence_status = "정상적인 수렴"
        else:
            convergence_status = "활발한 학습"
        
        result = {
            'dloss_depoch': dloss_depoch,
            'd2loss_depoch2': d2loss_depoch2,
            'abs_gradient': abs_gradient,
            'gradient_ratio': gradient_ratio,
            'final_gradient_magnitude': final_gradient_magnitude,
            'initial_gradient_magnitude': initial_gradient_magnitude,
            'convergence_status': convergence_status,
            'recent_variance': np.var(losses[-100:]) if len(losses) >= 100 else np.var(losses),
            'total_improvement': losses[0] - losses[-1] if len(losses) > 0 else 0,
            'loss_col': loss_col
        }
        
        # Validation loss 분석 (있는 경우)
        if 'val_loss' in epoch_df.columns:
            valid_indices = epoch_df['val_loss'].notna()
            if valid_indices.any():
                val_epochs = epoch_df.loc[valid_indices, epoch_col].values
                val_losses = epoch_df.loc[valid_indices, 'val_loss'].values
                
                if len(val_losses) > 1:
                    val_dloss_depoch = np.gradient(val_losses, val_epochs)
                    val_d2loss_depoch2 = np.gradient(val_dloss_depoch, val_epochs)
                    val_abs_gradient = np.abs(val_dloss_depoch)
                    
                    val_final_grad = np.mean(val_abs_gradient[-50:]) if len(val_abs_gradient) >= 50 else np.mean(val_abs_gradient[-10:])
                    val_initial_grad = np.mean(val_abs_gradient[:50]) if len(val_abs_gradient) >= 50 else np.mean(val_abs_gradient[:10])
                    val_gradient_ratio = val_final_grad / val_initial_grad if val_initial_grad > 0 else 0
                    
                    result.update({
                        'val_dloss_depoch': val_dloss_depoch,
                        'val_d2loss_depoch2': val_d2loss_depoch2,
                        'val_abs_gradient': val_abs_gradient,
                        'val_gradient_ratio': val_gradient_ratio,
                        'val_epochs': val_epochs,
                        'val_losses': val_losses,
                        'val_total_improvement': val_losses[0] - val_losses[-1]
                    })
        
        return result
    
    def generate_comprehensive_csv(self, experiment_id, epoch_df, batch_df, epoch_stats, gradient_analysis, output_dir):
        """종합 CSV 파일 생성"""
        csv_files = {}
        
        # Determine loss column name
        loss_col = gradient_analysis.get('loss_col', 'avg_loss')
        
        # CSV 전용 디렉토리 생성
        csv_dir = output_dir / "csv_data"
        csv_dir.mkdir(exist_ok=True)
        
        # 1. 기본 에폭 데이터
        epoch_enhanced = epoch_df.copy()
        epoch_enhanced['gradient'] = gradient_analysis['dloss_depoch']
        epoch_enhanced['gradient_2nd'] = gradient_analysis['d2loss_depoch2']
        epoch_enhanced['abs_gradient'] = gradient_analysis['abs_gradient']
        
        csv_files['epoch_data'] = csv_dir / f"epoch_comprehensive_{experiment_id}.csv"
        epoch_enhanced.to_csv(csv_files['epoch_data'], index=False, encoding='utf-8')
        
        # 2. 에폭 통계
        csv_files['epoch_statistics'] = csv_dir / f"epoch_statistics_{experiment_id}.csv"
        epoch_stats.to_csv(csv_files['epoch_statistics'], index=False, encoding='utf-8')
        
        # 3. 배치 데이터 (샘플링)
        if len(batch_df) > 10000:  # 너무 크면 샘플링
            batch_sample = batch_df.sample(n=10000, random_state=42).sort_values(['epoch', 'batch_num'])
        else:
            batch_sample = batch_df
        
        csv_files['batch_data'] = csv_dir / f"batch_data_{experiment_id}.csv"
        batch_sample.to_csv(csv_files['batch_data'], index=False, encoding='utf-8')
        
        # 4. 요약 통계
        summary_data = {
            'experiment_id': [experiment_id],
            'total_epochs': [len(epoch_df)],
            'total_batches': [len(batch_df)],
            'initial_loss': [epoch_df[loss_col].iloc[0]],
            'final_loss': [epoch_df[loss_col].iloc[-1]],
            'min_loss': [epoch_df[loss_col].min()],
            'gradient_ratio': [gradient_analysis['gradient_ratio']],
            'convergence_status': [gradient_analysis['convergence_status']],
            'total_improvement': [gradient_analysis['total_improvement']],
            'avg_batch_std': [epoch_stats['std'].mean()],
            'avg_batch_var': [epoch_stats['var'].mean()],
            'generation_time': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
        }
        
        summary_df = pd.DataFrame(summary_data)
        csv_files['summary'] = csv_dir / f"experiment_summary_{experiment_id}.csv"
        summary_df.to_csv(csv_files['summary'], index=False, encoding='utf-8')
        
        return csv_files
    
    def create_loss_plots(self, experiment_id, epoch_df, batch_df, output_dir):
        """손실 관련 그래프 생성"""
        # 그래프 전용 디렉토리 생성
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Determine loss column name
        loss_col = 'avg_loss' if 'avg_loss' in epoch_df.columns else 'loss'
        
        # 1. 에폭 평균 손실 (with optional validation loss)
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(epoch_df['epoch'], epoch_df[loss_col], 'b-', linewidth=2, marker='o', markersize=3, label='Training Loss')
        
        # Add validation loss if available
        if 'val_loss' in epoch_df.columns:
            # Filter out NaN values for validation loss
            valid_indices = epoch_df['val_loss'].notna()
            if valid_indices.any():
                ax.plot(epoch_df.loc[valid_indices, 'epoch'], 
                       epoch_df.loc[valid_indices, 'val_loss'], 
                       'r--', linewidth=2, marker='s', markersize=3, label='Validation Loss')
                ax.legend(fontsize=12)
        
        ax.set_title(f'Epoch Loss - {experiment_id}', fontsize=16, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=14)
        ax.set_ylabel('Loss', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        loss_plot_path = plots_dir / f"epoch_loss_{experiment_id}.png"
        plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 1-2. Validation Loss Only 그래프 (val_loss가 있는 경우)
        val_loss_plot_path = None
        if 'val_loss' in epoch_df.columns:
            valid_indices = epoch_df['val_loss'].notna()
            if valid_indices.any():
                fig, ax = plt.subplots(figsize=(12, 8))
                ax.plot(epoch_df.loc[valid_indices, 'epoch'], 
                       epoch_df.loc[valid_indices, 'val_loss'], 
                       'r-', linewidth=2, marker='s', markersize=4, label='Validation Loss')
                ax.set_title(f'Epoch Validation Loss - {experiment_id}', fontsize=16, fontweight='bold')
                ax.set_xlabel('Epoch', fontsize=14)
                ax.set_ylabel('Validation Loss', fontsize=14)
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=12)
                
                plt.tight_layout()
                val_loss_plot_path = plots_dir / f"epoch_val_loss_{experiment_id}.png"
                plt.savefig(val_loss_plot_path, dpi=300, bbox_inches='tight')
                plt.close()
        
        # 2. 선택된 에폭의 배치 손실
        total_epochs = epoch_df['epoch'].max()
        
        # 동적 에폭 선택
        if total_epochs <= 6:
            selected_epochs = list(range(1, total_epochs + 1))
        else:
            selected_epochs = [
                1,
                max(1, total_epochs // 10),
                max(1, total_epochs // 5),
                max(1, total_epochs * 3 // 10),
                max(1, total_epochs // 2),
                max(1, total_epochs * 8 // 10),
                total_epochs
            ]
            selected_epochs = sorted(list(set(selected_epochs)))
        
        # 실제 존재하는 에폭만 필터링
        available_epochs = batch_df['epoch'].unique()
        selected_epochs = [e for e in selected_epochs if e in available_epochs]
        
        if selected_epochs:
            fig, ax = plt.subplots(figsize=(15, 10))
            
            colors = plt.cm.tab10(np.linspace(0, 1, len(selected_epochs))) #type: ignore
            
            for i, epoch in enumerate(selected_epochs):
                epoch_batches = batch_df[batch_df['epoch'] == epoch]
                ax.plot(epoch_batches['batch_num'], epoch_batches['loss'], 
                       color=colors[i], linewidth=1.5, marker='o', markersize=2,
                       label=f'Epoch {epoch}', alpha=0.8)
            
            ax.set_title(f'Batch Loss Comparison - {experiment_id}', fontsize=16, fontweight='bold')
            ax.set_xlabel('Batch Number', fontsize=14)
            ax.set_ylabel('Loss', fontsize=14)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            batch_plot_path = plots_dir / f"batch_loss_{experiment_id}.png"
            plt.savefig(batch_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2-2. Validation Loss가 있는 에폭들의 배치 손실 그래프
        batch_val_plot_path = None
        if 'val_loss' in epoch_df.columns:
            valid_val_indices = epoch_df['val_loss'].notna()
            if valid_val_indices.any():
                val_epochs = epoch_df.loc[valid_val_indices, 'epoch'].values
                val_selected_epochs = [e for e in val_epochs if e in available_epochs]
                
                # 동적 에폭 선택 (validation epochs 중에서)
                total_val_epochs = len(val_selected_epochs)
                if total_val_epochs <= 6:
                    filtered_val_epochs = val_selected_epochs
                else:
                    # 균등하게 선택
                    indices = np.linspace(0, total_val_epochs - 1, min(7, total_val_epochs), dtype=int)
                    filtered_val_epochs = [val_selected_epochs[i] for i in indices]
                
                if filtered_val_epochs:
                    fig, ax = plt.subplots(figsize=(15, 10))
                    colors_val = plt.cm.Reds(np.linspace(0.4, 0.9, len(filtered_val_epochs))) #type: ignore
                    
                    for i, epoch in enumerate(filtered_val_epochs):
                        epoch_batches = batch_df[batch_df['epoch'] == epoch]
                        ax.plot(epoch_batches['batch_num'], epoch_batches['loss'], 
                               color=colors_val[i], linewidth=1.5, marker='s', markersize=2,
                               label=f'Epoch {epoch} (Val)', alpha=0.8)
                    
                    ax.set_title(f'Batch Loss Comparison (Validation Epochs) - {experiment_id}', fontsize=16, fontweight='bold')
                    ax.set_xlabel('Batch Number', fontsize=14)
                    ax.set_ylabel('Loss', fontsize=14)
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    ax.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    batch_val_plot_path = plots_dir / f"batch_loss_val_{experiment_id}.png"
                    plt.savefig(batch_val_plot_path, dpi=300, bbox_inches='tight')
                    plt.close()
        
        return loss_plot_path, batch_plot_path if selected_epochs else None
    
    def create_gradient_plots(self, experiment_id, epoch_df, gradient_analysis, output_dir):
        """기울기 분석 그래프 생성"""
        # 그래프 전용 디렉토리 사용
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Determine loss column name
        loss_col = gradient_analysis.get('loss_col', 'avg_loss')
        
        epochs = epoch_df['epoch'].values
        losses = epoch_df[loss_col].values
        dloss_depoch = gradient_analysis['dloss_depoch']
        d2loss_depoch2 = gradient_analysis['d2loss_depoch2']
        abs_gradient = gradient_analysis['abs_gradient']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 원본 Loss 곡선
        axes[0, 0].plot(epochs, losses, 'b-', linewidth=2)
        axes[0, 0].set_title('Loss vs Epoch')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 1차 미분 (기울기)
        axes[0, 1].plot(epochs, dloss_depoch, 'r-', linewidth=2)
        axes[0, 1].set_title('Loss Gradient (dLoss/dEpoch)')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Gradient')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        # 3. 기울기 크기 (절댓값, 로그 스케일)
        axes[1, 0].plot(epochs, abs_gradient, 'g-', linewidth=2)
        axes[1, 0].set_title('Absolute Gradient Magnitude')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('|Gradient|')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 2차 미분
        axes[1, 1].plot(epochs, d2loss_depoch2, 'm-', linewidth=2)
        axes[1, 1].set_title('Second Derivative (d²Loss/dEpoch²)')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Second Derivative')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        plt.suptitle(f'Gradient Analysis: {experiment_id}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        gradient_plot_path = plots_dir / f"gradient_analysis_{experiment_id}.png"
        plt.savefig(gradient_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Validation loss gradient plot (if available)
        val_gradient_plot_path = None
        if 'val_dloss_depoch' in gradient_analysis:
            val_epochs = gradient_analysis['val_epochs']
            val_losses = gradient_analysis['val_losses']
            val_dloss_depoch = gradient_analysis['val_dloss_depoch']
            val_d2loss_depoch2 = gradient_analysis['val_d2loss_depoch2']
            val_abs_gradient = gradient_analysis['val_abs_gradient']
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # 1. 원본 Validation Loss 곡선
            axes[0, 0].plot(val_epochs, val_losses, 'r-', linewidth=2)
            axes[0, 0].set_title('Validation Loss vs Epoch')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Validation Loss')
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. 1차 미분 (기울기)
            axes[0, 1].plot(val_epochs, val_dloss_depoch, 'orange', linewidth=2)
            axes[0, 1].set_title('Validation Loss Gradient (dLoss/dEpoch)')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Gradient')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
            
            # 3. 기울기 크기 (절댓값, 로그 스케일)
            axes[1, 0].plot(val_epochs, val_abs_gradient, 'purple', linewidth=2)
            axes[1, 0].set_title('Absolute Validation Gradient Magnitude')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('|Gradient|')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True, alpha=0.3)
            
            # 4. 2차 미분
            axes[1, 1].plot(val_epochs, val_d2loss_depoch2, 'brown', linewidth=2)
            axes[1, 1].set_title('Validation Second Derivative (d²Loss/dEpoch²)')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Second Derivative')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
            
            plt.suptitle(f'Validation Gradient Analysis: {experiment_id}', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            val_gradient_plot_path = plots_dir / f"gradient_analysis_val_{experiment_id}.png"
            plt.savefig(val_gradient_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        return gradient_plot_path
    
    def create_statistics_plots(self, experiment_id, epoch_stats, batch_df, output_dir, epoch_df=None):
        """통계 분석 그래프 생성"""
        # 그래프 전용 디렉토리 사용
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # 1. 에폭별 통계 트렌드
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 표준편차 트렌드
        axes[0, 0].plot(epoch_stats['epoch'], epoch_stats['std'], 'b-', linewidth=2)
        axes[0, 0].set_title('Standard Deviation per Epoch')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Standard Deviation')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 분산 트렌드
        axes[0, 1].plot(epoch_stats['epoch'], epoch_stats['var'], 'r-', linewidth=2)
        axes[0, 1].set_title('Variance per Epoch')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Variance')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 변동계수 트렌드
        axes[1, 0].plot(epoch_stats['epoch'], epoch_stats['cv'], 'g-', linewidth=2)
        axes[1, 0].set_title('Coefficient of Variation per Epoch')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('CV (std/mean)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 범위 트렌드
        axes[1, 1].plot(epoch_stats['epoch'], epoch_stats['range'], 'm-', linewidth=2)
        axes[1, 1].set_title('Loss Range per Epoch')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Range (max - min)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'Statistics Trends: {experiment_id}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        stats_plot_path = plots_dir / f"statistics_trends_{experiment_id}.png"
        plt.savefig(stats_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 통계 분포 히스토그램
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 표준편차 분포
        axes[0, 0].hist(epoch_stats['std'], bins=30, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 0].set_title('Distribution of Standard Deviations')
        axes[0, 0].set_xlabel('Standard Deviation')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 분산 분포
        axes[0, 1].hist(epoch_stats['var'], bins=30, alpha=0.7, color='red', edgecolor='black')
        axes[0, 1].set_title('Distribution of Variances')
        axes[0, 1].set_xlabel('Variance')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 변동계수 분포
        axes[1, 0].hist(epoch_stats['cv'], bins=30, alpha=0.7, color='green', edgecolor='black')
        axes[1, 0].set_title('Distribution of Coefficient of Variation')
        axes[1, 0].set_xlabel('CV (std/mean)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 상관관계 히트맵
        corr_cols = ['mean', 'std', 'var', 'cv', 'range']
        correlation_matrix = epoch_stats[corr_cols].corr()
        im = axes[1, 1].imshow(correlation_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        axes[1, 1].set_xticks(range(len(corr_cols)))
        axes[1, 1].set_yticks(range(len(corr_cols)))
        axes[1, 1].set_xticklabels(corr_cols, rotation=45)
        axes[1, 1].set_yticklabels(corr_cols)
        axes[1, 1].set_title('Correlation Matrix')
        
        # 상관관계 값 표시
        for i in range(len(corr_cols)):
            for j in range(len(corr_cols)):
                text = axes[1, 1].text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                                     ha="center", va="center", color="black", fontsize=8)
        
        plt.colorbar(im, ax=axes[1, 1], shrink=0.8)
        plt.suptitle(f'Statistics Distributions: {experiment_id}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        dist_plot_path = plots_dir / f"statistics_distributions_{experiment_id}.png"
        plt.savefig(dist_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Validation Loss 분포 분석 그래프 (val_loss가 있는 경우)
        val_dist_plot_path = None
        if epoch_df is not None and 'val_loss' in epoch_df.columns:
            valid_indices = epoch_df['val_loss'].notna()
            if valid_indices.any():
                val_losses = epoch_df.loc[valid_indices, 'val_loss'].values
                
                fig, axes = plt.subplots(2, 2, figsize=(16, 12))
                
                # Validation Loss 분포
                axes[0, 0].hist(val_losses, bins=20, alpha=0.7, color='red', edgecolor='black')
                axes[0, 0].set_title('Distribution of Validation Loss')
                axes[0, 0].set_xlabel('Validation Loss')
                axes[0, 0].set_ylabel('Frequency')
                axes[0, 0].grid(True, alpha=0.3)
                
                # Training Loss와 Validation Loss 비교
                train_loss_col = 'avg_loss' if 'avg_loss' in epoch_df.columns else 'loss'
                train_losses = epoch_df[train_loss_col].values
                axes[0, 1].hist([train_losses, val_losses], label=['Training', 'Validation'], 
                               color=['blue', 'red'], alpha=0.7, edgecolor='black', bins=15)
                axes[0, 1].set_title('Train vs Validation Loss Distribution')
                axes[0, 1].set_xlabel('Loss')
                axes[0, 1].set_ylabel('Frequency')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
                
                # Validation Loss Box Plot
                axes[1, 0].boxplot([train_losses, val_losses], labels=['Training', 'Validation'])
                axes[1, 0].set_title('Box Plot: Train vs Validation Loss')
                axes[1, 0].set_ylabel('Loss')
                axes[1, 0].grid(True, alpha=0.3, axis='y')
                
                # Validation Loss 통계
                val_mean = np.mean(val_losses)
                val_std = np.std(val_losses)
                val_min = np.min(val_losses)
                val_max = np.max(val_losses)
                train_mean = np.mean(train_losses)
                train_std = np.std(train_losses)
                
                stats_text = f"Validation Loss Statistics:\n"
                stats_text += f"Mean: {val_mean:.6f}\n"
                stats_text += f"Std: {val_std:.6f}\n"
                stats_text += f"Min: {val_min:.6f}\n"
                stats_text += f"Max: {val_max:.6f}\n\n"
                stats_text += f"Training Loss Statistics:\n"
                stats_text += f"Mean: {train_mean:.6f}\n"
                stats_text += f"Std: {train_std:.6f}"
                
                axes[1, 1].text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center',
                               family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                axes[1, 1].axis('off')
                
                plt.suptitle(f'Validation Loss Statistics: {experiment_id}', fontsize=16, fontweight='bold')
                plt.tight_layout()
                
                val_dist_plot_path = plots_dir / f"statistics_distributions_val_{experiment_id}.png"
                plt.savefig(val_dist_plot_path, dpi=300, bbox_inches='tight')
                plt.close()
        
        return stats_plot_path, dist_plot_path
    
    def generate_experiment_report(self, experiment_id, epoch_df, batch_df, epoch_stats, gradient_analysis, csv_files, plot_files, output_dir):
        """실험 리포트 생성"""
        report_path = output_dir / f"experiment_report_{experiment_id}.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# 실험 리포트: {experiment_id}\n\n")
            f.write(f"**생성 일시**: {datetime.now().strftime('%Y년 %m월 %d일 %H시 %M분')}\n\n")
            
            # 실험 개요
            f.write("## 📊 실험 개요\n\n")
            f.write(f"- **총 에폭 수**: {len(epoch_df)}\n")
            f.write(f"- **총 배치 수**: {len(batch_df)}\n")
            f.write(f"- **초기 손실**: {epoch_df['avg_loss'].iloc[0]:.6f}\n")
            f.write(f"- **최종 손실**: {epoch_df['avg_loss'].iloc[-1]:.6f}\n")
            f.write(f"- **최소 손실**: {epoch_df['avg_loss'].min():.6f}\n")
            f.write(f"- **총 개선량**: {gradient_analysis['total_improvement']:.6f}\n\n")
            
            # 기울기 분석
            f.write("## 🔍 기울기 분석\n\n")
            f.write(f"- **기울기 비율 (최종/초기)**: {gradient_analysis['gradient_ratio']:.4f}\n")
            f.write(f"- **수렴 상태**: {gradient_analysis['convergence_status']}\n")
            f.write(f"- **초기 기울기 크기**: {gradient_analysis['initial_gradient_magnitude']:.8f}\n")
            f.write(f"- **최종 기울기 크기**: {gradient_analysis['final_gradient_magnitude']:.8f}\n")
            f.write(f"- **최근 분산**: {gradient_analysis['recent_variance']:.8f}\n\n")
            
            # 통계 요약
            f.write("## 📈 배치 손실 통계\n\n")
            f.write(f"- **평균 표준편차**: {epoch_stats['std'].mean():.6f}\n")
            f.write(f"- **평균 분산**: {epoch_stats['var'].mean():.6f}\n")
            f.write(f"- **평균 변동계수**: {epoch_stats['cv'].mean():.6f}\n")
            f.write(f"- **표준편차 범위**: {epoch_stats['std'].min():.6f} ~ {epoch_stats['std'].max():.6f}\n\n")
            
            # 생성된 파일들
            f.write("## 📁 생성된 파일들\n\n")
            f.write("### CSV 데이터 (csv_data/ 폴더)\n")
            for file_type, file_path in csv_files.items():
                f.write(f"- **{file_type}**: `csv_data/{file_path.name}`\n")
            
            f.write("\n### 그래프 (plots/ 폴더)\n")
            for plot_desc, plot_path in plot_files.items():
                if plot_path:
                    f.write(f"- **{plot_desc}**: `plots/{plot_path.name}`\n")
            
            f.write(f"\n---\n")
            f.write(f"*리포트 생성: Unified Model Monitor v1.0*\n")
        
        return report_path
    
    def process_experiment(self, experiment):
        """단일 실험 처리"""
        experiment_id = experiment['id']
        print(f"\n{'='*60}")
        print(f"📊 실험 처리 중: {experiment_id}")
        print(f"{'='*60}")
        
        # 출력 디렉토리 생성
        output_dir = self.output_base_dir / experiment_id
        output_dir.mkdir(exist_ok=True)
        
        # 데이터 로드
        print("📂 데이터 로드 중...")
        epoch_df, batch_df = self.load_experiment_data(experiment)
        if epoch_df is None or batch_df is None:
            return None
        
        print(f"   - 에폭 데이터: {len(epoch_df)} 개")
        print(f"   - 배치 데이터: {len(batch_df)} 개")
        
        # 통계 계산
        print("🔢 통계 계산 중...")
        epoch_stats = self.calculate_epoch_statistics(batch_df)
        gradient_analysis = self.calculate_gradient_analysis(epoch_df)
        
        # CSV 파일 생성
        print("💾 CSV 파일 생성 중...")
        csv_files = self.generate_comprehensive_csv(
            experiment_id, epoch_df, batch_df, epoch_stats, gradient_analysis, output_dir
        )
        
        # 그래프 생성
        print("📈 그래프 생성 중...")
        
        # 손실 그래프
        loss_plot, batch_plot = self.create_loss_plots(experiment_id, epoch_df, batch_df, output_dir)
        # Check if validation loss graph was created
        plots_dir = output_dir / "plots"
        val_loss_plot = plots_dir / f"epoch_val_loss_{experiment_id}.png"
        batch_val_plot = plots_dir / f"batch_loss_val_{experiment_id}.png"
        if val_loss_plot.exists():
            print(f"   ✅ Training + Validation Loss 그래프 생성됨")
            print(f"   ✅ Validation Loss Only 그래프 생성됨")
        if batch_val_plot.exists():
            print(f"   ✅ Batch Loss (Train) 그래프 생성됨")
            print(f"   ✅ Batch Loss (Validation) 그래프 생성됨")
        
        # 기울기 그래프  
        gradient_plot = self.create_gradient_plots(experiment_id, epoch_df, gradient_analysis, output_dir)
        # Check if validation gradient plot was created
        val_gradient_plot = plots_dir / f"gradient_analysis_val_{experiment_id}.png"
        if val_gradient_plot.exists():
            print(f"   ✅ Training Gradient 분석 생성됨")
            print(f"   ✅ Validation Gradient 분석 생성됨")
        else:
            print(f"   ✅ Training Gradient 분석 생성됨")
        
        # 통계 그래프 (배치가 충분할 때만)
        avg_batches_per_epoch = epoch_stats['count'].mean()
        if avg_batches_per_epoch > 2:
            stats_plot, dist_plot = self.create_statistics_plots(experiment_id, epoch_stats, batch_df, output_dir, epoch_df)
            # Check if validation statistics distribution plot was created
            val_dist_plot = plots_dir / f"statistics_distributions_val_{experiment_id}.png"
            if val_dist_plot.exists():
                print(f"   ✅ Statistics Trends (Train) 그래프 생성됨")
                print(f"   ✅ Statistics Distributions (Train) 그래프 생성됨")
                print(f"   ✅ Statistics Distributions (Validation) 그래프 생성됨")
            else:
                print(f"   ✅ Statistics Trends (Train) 그래프 생성됨")
                print(f"   ✅ Statistics Distributions (Train) 그래프 생성됨")
        else:
            print(f"⚠️  배치 수 부족 (평균 {avg_batches_per_epoch:.1f}개/에폭) - 통계 그래프 생략")
            stats_plot, dist_plot = None, None
            # 배치 손실 비교 그래프도 생략
            batch_plot = None
        
        plot_files = {
            "에폭 평균 손실": loss_plot,
            "배치 손실 비교": batch_plot,
            "기울기 분석": gradient_plot,
        }
        if stats_plot:
            plot_files["통계 트렌드"] = stats_plot
        if dist_plot:
            plot_files["통계 분포"] = dist_plot
        
        # 실험 리포트 생성
        print("📝 실험 리포트 생성 중...")
        report_path = self.generate_experiment_report(
            experiment_id, epoch_df, batch_df, epoch_stats, gradient_analysis, csv_files, plot_files, output_dir
        )
        
        print(f"✅ 완료! 결과 저장: {output_dir}")
        print(f"📋 리포트: {report_path.name}")
        
        return {
            'experiment_id': experiment_id,
            'output_dir': output_dir,
            'epoch_df': epoch_df,
            'batch_df': batch_df,
            'epoch_stats': epoch_stats,
            'gradient_analysis': gradient_analysis,
            'csv_files': csv_files,
            'plot_files': plot_files,
            'report_path': report_path
        }
    
    def generate_comparison_report(self, processed_experiments):
        """전체 실험 비교 리포트 생성"""
        if not processed_experiments:
            return
        
        print(f"\n{'='*80}")
        print("🏆 전체 실험 비교 리포트 생성")
        print(f"{'='*80}")
        
        # 비교 데이터 수집
        comparison_data = []
        for exp in processed_experiments:
            comparison_data.append({
                'experiment_id': exp['experiment_id'],
                'total_epochs': len(exp['epoch_df']),
                'initial_loss': exp['epoch_df']['avg_loss'].iloc[0],
                'final_loss': exp['epoch_df']['avg_loss'].iloc[-1],
                'min_loss': exp['epoch_df']['avg_loss'].min(),
                'total_improvement': exp['gradient_analysis']['total_improvement'],
                'gradient_ratio': exp['gradient_analysis']['gradient_ratio'],
                'convergence_status': exp['gradient_analysis']['convergence_status'],
                'avg_batch_std': exp['epoch_stats']['std'].mean(),
                'avg_batch_var': exp['epoch_stats']['var'].mean(),
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # 통합 CSV 디렉토리 생성
        consolidated_csv_dir = self.output_base_dir / "consolidated_csv"
        consolidated_csv_dir.mkdir(exist_ok=True)
        
        # 비교 CSV 저장
        comparison_csv = consolidated_csv_dir / "experiments_comparison.csv"
        comparison_df.to_csv(comparison_csv, index=False, encoding='utf-8')
        
        # 모든 실험의 CSV 데이터를 통합
        self.consolidate_all_csv_data(processed_experiments, consolidated_csv_dir)
        
        # 비교 리포트 생성
        report_path = self.output_base_dir / "comparison_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# 전체 실험 비교 리포트\n\n")
            f.write(f"**생성 일시**: {datetime.now().strftime('%Y년 %m월 %d일 %H시 %M분')}\n")
            f.write(f"**분석 실험 수**: {len(processed_experiments)}\n\n")
            
            # 기울기 소실 기준 정렬
            sorted_by_gradient = comparison_df.sort_values('total_improvement')
            
            f.write("## 🎯 실험 성능 순위 (기울기 소실 기준)\n\n")
            f.write("| 순위 | 실험 ID | 기울기 비율 | 수렴 상태 | 총 개선량 |\n")
            f.write("|------|---------|-------------|-----------|----------|\n")
            
            for i, (_, row) in enumerate(sorted_by_gradient.iterrows(), 1):
                f.write(f"| {i} | {row['experiment_id']} | {row['gradient_ratio']:.4f} | {row['convergence_status']} | {row['total_improvement']:.6f} |\n")
            
            # 최고/최악 실험
            best_exp = sorted_by_gradient.iloc[-1]  # 최종 개선량이 좋은 것
            worst_exp = sorted_by_gradient.iloc[0]   # 그 반대
            
            f.write(f"\n## 🏆 최고 성능 실험\n\n")
            f.write(f"**실험 ID**: {best_exp['experiment_id']}\n")
            f.write(f"- 기울기 비율: {best_exp['gradient_ratio']:.4f}\n")
            f.write(f"- 수렴 상태: {best_exp['convergence_status']}\n")
            f.write(f"- 총 개선량: {best_exp['total_improvement']:.6f}\n")
            
            f.write(f"\n## ⚠️ 개선 필요 실험\n\n")
            f.write(f"**실험 ID**: {worst_exp['experiment_id']}\n")
            f.write(f"- 기울기 비율: {worst_exp['gradient_ratio']:.4f}\n")
            f.write(f"- 수렴 상태: {worst_exp['convergence_status']}\n")
            f.write(f"- 총 개선량: {worst_exp['total_improvement']:.6f}\n")
            
            f.write(f"\n## 📊 전체 통계\n\n")
            f.write(f"- **평균 기울기 비율**: {comparison_df['gradient_ratio'].mean():.4f}\n")
            f.write(f"- **평균 총 개선량**: {comparison_df['total_improvement'].mean():.6f}\n")
            f.write(f"- **평균 최종 손실**: {comparison_df['final_loss'].mean():.6f}\n")
            
            f.write(f"\n## 📁 통합 CSV 파일\n\n")
            f.write(f"모든 실험의 CSV 데이터가 `consolidated_csv/` 폴더에 통합되었습니다:\n")
            f.write(f"- **실험 비교**: `experiments_comparison.csv`\n")
            f.write(f"- **모든 에폭 데이터**: `all_epochs_comprehensive.csv`\n")
            f.write(f"- **모든 에폭 통계**: `all_epochs_statistics.csv`\n")
            f.write(f"- **모든 실험 요약**: `all_experiments_summary.csv`\n")
        
        print(f"✅ 비교 리포트 생성: {report_path}")
        print(f"✅ 비교 CSV 생성: {comparison_csv}")
        print(f"✅ 통합 CSV 생성: {consolidated_csv_dir}")
    
    def consolidate_all_csv_data(self, processed_experiments, output_dir):
        """모든 실험의 CSV 데이터를 통합"""
        all_epochs_data = []
        all_stats_data = []
        all_summary_data = []
        
        for exp in processed_experiments:
            experiment_id = exp['experiment_id']
            
            # 에폭 데이터 통합
            epoch_enhanced = exp['epoch_df'].copy()
            epoch_enhanced['gradient'] = exp['gradient_analysis']['dloss_depoch']
            epoch_enhanced['gradient_2nd'] = exp['gradient_analysis']['d2loss_depoch2']
            epoch_enhanced['abs_gradient'] = exp['gradient_analysis']['abs_gradient']
            epoch_enhanced['experiment_id'] = experiment_id
            all_epochs_data.append(epoch_enhanced)
            
            # 통계 데이터 통합
            stats_data = exp['epoch_stats'].copy()
            stats_data['experiment_id'] = experiment_id
            all_stats_data.append(stats_data)
            
            # 요약 데이터는 이미 수집되어 있음
        
        # 통합 DataFrame 생성 및 저장
        if all_epochs_data:
            all_epochs_df = pd.concat(all_epochs_data, ignore_index=True)
            all_epochs_csv = output_dir / "all_epochs_comprehensive.csv"
            all_epochs_df.to_csv(all_epochs_csv, index=False, encoding='utf-8')
            print(f"✅ 통합 에폭 데이터: {all_epochs_csv}")
        
        if all_stats_data:
            all_stats_df = pd.concat(all_stats_data, ignore_index=True)
            all_stats_csv = output_dir / "all_epochs_statistics.csv"
            all_stats_df.to_csv(all_stats_csv, index=False, encoding='utf-8')
            print(f"✅ 통합 통계 데이터: {all_stats_csv}")
        
        # 요약 데이터 통합 (개별 요약 파일들을 읽어서 통합)
        all_summary_data = []
        for exp in processed_experiments:
            summary_file = exp['csv_files']['summary']
            if summary_file.exists():
                summary_df = pd.read_csv(summary_file)
                all_summary_data.append(summary_df)
        
        if all_summary_data:
            all_summary_df = pd.concat(all_summary_data, ignore_index=True)
            all_summary_csv = output_dir / "all_experiments_summary.csv"
            all_summary_df.to_csv(all_summary_csv, index=False, encoding='utf-8')
            print(f"✅ 통합 요약 데이터: {all_summary_csv}")
    
    def run(self, experiment_ids=None, latest_only=False):
        """메인 실행 함수"""
        print("🚀 통합 모델 모니터링 시스템 시작")
        print(f"📂 작업 디렉토리: {self.workspace_dir}")
        print(f"💾 출력 디렉토리: {self.output_base_dir}")
        
        # 실험 발견
        experiments = self.discover_experiments()
        if not experiments:
            print("❌ 분석할 실험이 없습니다.")
            return
        
        # 실험 필터링
        if latest_only:
            experiments = experiments[:1]
            print(f"🎯 최신 실험만 처리: {experiments[0]['id']}")
        elif experiment_ids:
            experiments = [exp for exp in experiments if exp['id'] in experiment_ids]
            print(f"🎯 지정된 실험만 처리: {[exp['id'] for exp in experiments]}")
        
        if not experiments:
            print("❌ 처리할 실험이 없습니다.")
            return
        
        # 각 실험 처리
        processed_experiments = []
        for experiment in experiments:
            try:
                result = self.process_experiment(experiment)
                if result:
                    processed_experiments.append(result)
            except Exception as e:
                print(f"❌ {experiment['id']} 처리 실패: {e}")
        
        # 전체 비교 리포트 생성
        if len(processed_experiments) > 1:
            self.generate_comparison_report(processed_experiments)
        
        print(f"\n🎉 모든 처리 완료!")
        print(f"📁 결과 위치: {self.output_base_dir}")
        print(f"✅ 처리된 실험: {len(processed_experiments)}개")

def main():
    parser = argparse.ArgumentParser(description='통합 모델 모니터링 시스템')
    parser.add_argument('--workspace', default=os.getcwd(),
                       help='작업 디렉토리 경로 (기본값: 현재 디렉토리)')
    parser.add_argument('--output-dir', default='model_result_monitoring',
                       help='출력 디렉토리 이름 (기본값: model_result_monitoring)')
    parser.add_argument('--experiments', nargs='+',
                       help='처리할 특정 실험 ID들 (기본값: 모든 실험)')
    parser.add_argument('--latest-only', action='store_true',
                       help='최신 실험만 처리')
    
    args = parser.parse_args()
    
    # 모니터 생성 및 실행
    monitor = ModelMonitor(workspace_dir=args.workspace, output_base_dir=args.output_dir)
    monitor.run(experiment_ids=args.experiments, latest_only=args.latest_only)

if __name__ == "__main__":
    main()
