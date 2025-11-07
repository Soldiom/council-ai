"""
📊 DATA ANALYTICS & MONITORING SYSTEM

Tracks and analyzes:
- Training data collection (daily/weekly/monthly)
- Model performance metrics
- Data quality scores
- Cost tracking
- Progress reports
- Forensic data specific analytics
"""

import json
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import statistics


@dataclass
class DataCollectionMetrics:
    """Metrics for data collection"""
    timestamp: str
    date: str
    hour: int
    
    # Collection stats
    ensemble_examples: int = 0
    forensic_examples: int = 0
    deepfake_examples: int = 0
    document_examples: int = 0
    audio_examples: int = 0  # NEW: Whisper, VoxCeleb data
    image_examples: int = 0  # NEW: Forensic images
    
    # Model usage
    models_used: List[str] = None
    models_count: int = 0
    
    # Quality metrics
    avg_response_length: float = 0.0
    avg_quality_score: float = 0.0
    data_diversity_score: float = 0.0
    
    # Cost tracking
    estimated_cost: float = 0.0
    
    def __post_init__(self):
        if self.models_used is None:
            self.models_used = []


@dataclass
class ModelPerformanceMetrics:
    """Track model training performance"""
    timestamp: str
    model_name: str
    
    # Training metrics
    training_examples: int = 0
    training_time_minutes: float = 0.0
    epochs: int = 3
    final_loss: float = 0.0
    
    # Quality metrics
    perplexity: float = 0.0
    bleu_score: float = 0.0
    rouge_score: float = 0.0
    
    # Deployment
    deployed_to_hf: bool = False
    model_size_mb: float = 0.0


class DataAnalytics:
    """Comprehensive data analytics and monitoring"""
    
    def __init__(self, db_path: str = "training_data/analytics.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_database()
        
    def init_database(self):
        """Initialize SQLite database for analytics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Data collection metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS data_collection (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                date TEXT,
                hour INTEGER,
                ensemble_examples INTEGER,
                forensic_examples INTEGER,
                deepfake_examples INTEGER,
                document_examples INTEGER,
                audio_examples INTEGER,
                image_examples INTEGER,
                models_used TEXT,
                models_count INTEGER,
                avg_response_length REAL,
                avg_quality_score REAL,
                data_diversity_score REAL,
                estimated_cost REAL
            )
        ''')
        
        # Model performance table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                model_name TEXT,
                training_examples INTEGER,
                training_time_minutes REAL,
                epochs INTEGER,
                final_loss REAL,
                perplexity REAL,
                bleu_score REAL,
                rouge_score REAL,
                deployed_to_hf BOOLEAN,
                model_size_mb REAL
            )
        ''')
        
        # Forensic-specific analytics
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS forensic_analytics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                data_type TEXT,
                source TEXT,
                quality_score REAL,
                file_size_mb REAL,
                duration_seconds REAL,
                metadata TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def log_collection(self, metrics: DataCollectionMetrics):
        """Log data collection metrics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO data_collection VALUES (
                NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        ''', (
            metrics.timestamp,
            metrics.date,
            metrics.hour,
            metrics.ensemble_examples,
            metrics.forensic_examples,
            metrics.deepfake_examples,
            metrics.document_examples,
            metrics.audio_examples,
            metrics.image_examples,
            json.dumps(metrics.models_used),
            metrics.models_count,
            metrics.avg_response_length,
            metrics.avg_quality_score,
            metrics.data_diversity_score,
            metrics.estimated_cost
        ))
        
        conn.commit()
        conn.close()
    
    def log_training(self, metrics: ModelPerformanceMetrics):
        """Log model training metrics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO model_performance VALUES (
                NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        ''', (
            metrics.timestamp,
            metrics.model_name,
            metrics.training_examples,
            metrics.training_time_minutes,
            metrics.epochs,
            metrics.final_loss,
            metrics.perplexity,
            metrics.bleu_score,
            metrics.rouge_score,
            metrics.deployed_to_hf,
            metrics.model_size_mb
        ))
        
        conn.commit()
        conn.close()
    
    def log_forensic_data(self, data_type: str, source: str, quality_score: float, 
                         file_size_mb: float = 0.0, duration_seconds: float = 0.0, 
                         metadata: Dict = None):
        """Log forensic-specific data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO forensic_analytics VALUES (
                NULL, ?, ?, ?, ?, ?, ?, ?
            )
        ''', (
            datetime.now().isoformat(),
            data_type,
            source,
            quality_score,
            file_size_mb,
            duration_seconds,
            json.dumps(metadata or {})
        ))
        
        conn.commit()
        conn.close()
    
    def get_daily_report(self, date: str = None) -> Dict[str, Any]:
        """Generate daily report"""
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get daily stats
        cursor.execute('''
            SELECT 
                SUM(ensemble_examples),
                SUM(forensic_examples),
                SUM(deepfake_examples),
                SUM(document_examples),
                SUM(audio_examples),
                SUM(image_examples),
                AVG(models_count),
                AVG(avg_quality_score),
                SUM(estimated_cost)
            FROM data_collection
            WHERE date = ?
        ''', (date,))
        
        row = cursor.fetchone()
        conn.close()
        
        return {
            "date": date,
            "total_examples": sum(row[:6]) if row[0] else 0,
            "ensemble": row[0] or 0,
            "forensic": row[1] or 0,
            "deepfake": row[2] or 0,
            "document": row[3] or 0,
            "audio": row[4] or 0,
            "image": row[5] or 0,
            "avg_models_used": round(row[6], 1) if row[6] else 0,
            "avg_quality": round(row[7], 2) if row[7] else 0,
            "total_cost": round(row[8], 2) if row[8] else 0
        }
    
    def get_weekly_report(self) -> Dict[str, Any]:
        """Generate weekly report"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                date,
                SUM(ensemble_examples + forensic_examples + deepfake_examples + 
                    document_examples + audio_examples + image_examples) as total
            FROM data_collection
            WHERE date >= ? AND date <= ?
            GROUP BY date
            ORDER BY date
        ''', (start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')))
        
        daily_totals = cursor.fetchall()
        conn.close()
        
        return {
            "period": "7 days",
            "start_date": start_date.strftime('%Y-%m-%d'),
            "end_date": end_date.strftime('%Y-%m-%d'),
            "daily_breakdown": [{"date": d[0], "examples": d[1]} for d in daily_totals],
            "total_examples": sum(d[1] for d in daily_totals),
            "avg_per_day": round(statistics.mean([d[1] for d in daily_totals]), 1) if daily_totals else 0
        }
    
    def get_monthly_report(self) -> Dict[str, Any]:
        """Generate monthly report"""
        today = datetime.now()
        start_date = today.replace(day=1)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                SUM(ensemble_examples),
                SUM(forensic_examples),
                SUM(deepfake_examples),
                SUM(document_examples),
                SUM(audio_examples),
                SUM(image_examples),
                SUM(estimated_cost),
                COUNT(DISTINCT date)
            FROM data_collection
            WHERE date >= ?
        ''', (start_date.strftime('%Y-%m-%d'),))
        
        row = cursor.fetchone()
        
        # Get model training stats
        cursor.execute('''
            SELECT 
                COUNT(*),
                AVG(final_loss),
                SUM(training_examples)
            FROM model_performance
            WHERE timestamp >= ?
        ''', (start_date.isoformat(),))
        
        training_row = cursor.fetchone()
        conn.close()
        
        total_examples = sum(row[:6]) if row[0] else 0
        days_active = row[7] or 1
        
        return {
            "month": today.strftime('%Y-%m'),
            "total_examples": total_examples,
            "by_type": {
                "ensemble": row[0] or 0,
                "forensic": row[1] or 0,
                "deepfake": row[2] or 0,
                "document": row[3] or 0,
                "audio": row[4] or 0,
                "image": row[5] or 0
            },
            "total_cost": round(row[6], 2) if row[6] else 0,
            "days_active": days_active,
            "avg_per_day": round(total_examples / days_active, 1),
            "models_trained": training_row[0] or 0,
            "avg_loss": round(training_row[1], 4) if training_row[1] else 0,
            "total_training_examples": training_row[2] or 0
        }
    
    def print_daily_summary(self, date: str = None):
        """Print daily summary"""
        report = self.get_daily_report(date)
        
        print("\n" + "=" * 70)
        print(f"📊 DAILY REPORT - {report['date']}")
        print("=" * 70)
        print(f"\n📈 Data Collection:")
        print(f"   Total Examples: {report['total_examples']}")
        print(f"   ├─ Ensemble: {report['ensemble']}")
        print(f"   ├─ Forensic: {report['forensic']}")
        print(f"   ├─ Deepfake: {report['deepfake']}")
        print(f"   ├─ Document: {report['document']}")
        print(f"   ├─ Audio (Whisper/VoxCeleb): {report['audio']}")
        print(f"   └─ Images (Forensic): {report['image']}")
        print(f"\n🤖 Models:")
        print(f"   Average Models Used: {report['avg_models_used']}")
        print(f"   Average Quality Score: {report['avg_quality']}/10")
        print(f"\n💰 Cost:")
        print(f"   Estimated: ${report['total_cost']}")
        print("\n" + "=" * 70 + "\n")
    
    def print_weekly_summary(self):
        """Print weekly summary"""
        report = self.get_weekly_report()
        
        print("\n" + "=" * 70)
        print(f"📊 WEEKLY REPORT - {report['start_date']} to {report['end_date']}")
        print("=" * 70)
        print(f"\n📈 Total Examples: {report['total_examples']}")
        print(f"   Average per day: {report['avg_per_day']}")
        print(f"\n📅 Daily Breakdown:")
        for day in report['daily_breakdown']:
            print(f"   {day['date']}: {day['examples']} examples")
        print("\n" + "=" * 70 + "\n")
    
    def print_monthly_summary(self):
        """Print monthly summary"""
        report = self.get_monthly_report()
        
        print("\n" + "=" * 70)
        print(f"📊 MONTHLY REPORT - {report['month']}")
        print("=" * 70)
        print(f"\n📈 Data Collection:")
        print(f"   Total Examples: {report['total_examples']}")
        print(f"   Average per day: {report['avg_per_day']}")
        print(f"\n📁 By Type:")
        for dtype, count in report['by_type'].items():
            print(f"   {dtype.capitalize()}: {count}")
        print(f"\n🎓 Model Training:")
        print(f"   Models Trained: {report['models_trained']}")
        print(f"   Training Examples Used: {report['total_training_examples']}")
        print(f"   Average Loss: {report['avg_loss']}")
        print(f"\n💰 Total Cost: ${report['total_cost']}")
        print(f"📅 Days Active: {report['days_active']}")
        print("\n" + "=" * 70 + "\n")


# Global analytics instance
_analytics = None

def get_analytics() -> DataAnalytics:
    """Get global analytics instance"""
    global _analytics
    if _analytics is None:
        _analytics = DataAnalytics()
    return _analytics


if __name__ == "__main__":
    # Test analytics
    analytics = DataAnalytics()
    
    # Simulate some data
    metrics = DataCollectionMetrics(
        timestamp=datetime.now().isoformat(),
        date=datetime.now().strftime('%Y-%m-%d'),
        hour=datetime.now().hour,
        ensemble_examples=50,
        forensic_examples=20,
        audio_examples=15,
        image_examples=10,
        models_used=["gpt-4o", "claude-3.5-sonnet", "whisper-large-v3"],
        models_count=3,
        avg_quality_score=8.5,
        estimated_cost=0.25
    )
    
    analytics.log_collection(metrics)
    analytics.print_daily_summary()
