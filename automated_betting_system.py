#!/usr/bin/env python3
"""
Automated Bet Staking System for Bettensor MLB Predictions 
This system automatically places bets based on your ML predictions
"""

import requests
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import asyncio
from dataclasses import dataclass

@dataclass
class BettingAccount:
    """Betting account configuration"""
    platform: str
    api_key: str
    api_secret: str
    base_url: str
    balance: float = 0.0
    max_daily_stake: float = 500.0
    max_bet_percentage: float = 0.05  # 5% of balance per bet

@dataclass
class AutoBet:
    """Automated bet configuration"""
    game_id: str
    home_team: str
    away_team: str
    predicted_outcome: str
    confidence: float
    recommended_stake: float
    odds: float
    sportsbook: str
    timestamp: datetime
    status: str = "pending"  # pending, placed, failed, cancelled

class AutomatedBettingEngine:
    """Automated betting engine that stakes bets based on ML predictions"""
    
    def __init__(self, betting_accounts: List[BettingAccount], min_confidence: float = 0.65):
        self.betting_accounts = betting_accounts
        self.min_confidence = min_confidence
        self.daily_stakes = {}  # Track daily betting amounts per account
        self.pending_bets = []
        self.placed_bets = []
        
        # Risk management settings
        self.max_concurrent_bets = 10
        self.max_stake_per_bet = 200.0
        self.blackout_hours = [1, 2, 3, 4, 5]  # Don't bet during these hours
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def should_place_bet(self, prediction: Dict, account: BettingAccount) -> bool:
        """Determine if we should automatically place this bet"""
        
        # Check confidence threshold
        if prediction['confidence'] < self.min_confidence:
            self.logger.info(f"❌ Skipping bet - confidence {prediction['confidence']:.1%} below threshold {self.min_confidence:.1%}")
            return False
        
        # Check if it's blackout hours
        current_hour = datetime.now().hour
        if current_hour in self.blackout_hours:
            self.logger.info(f"❌ Skipping bet - blackout hour {current_hour}")
            return False
        
        # Check daily stake limits
        today = datetime.now().date()
        daily_key = f"{account.platform}_{today}"
        daily_staked = self.daily_stakes.get(daily_key, 0)
        
        if daily_staked >= account.max_daily_stake:
            self.logger.info(f"❌ Skipping bet - daily limit reached (${daily_staked:.2f})")
            return False
        
        # Check if stake amount is reasonable
        if prediction['recommended_stake'] > self.max_stake_per_bet:
            self.logger.info(f"❌ Skipping bet - stake too large (${prediction['recommended_stake']:.2f})")
            return False
        
        # Check if we have enough balance
        if prediction['recommended_stake'] > account.balance * account.max_bet_percentage:
            self.logger.info(f"❌ Skipping bet - stake exceeds percentage limit")
            return False
        
        # Check concurrent bets
        if len(self.pending_bets) >= self.max_concurrent_bets:
            self.logger.info(f"❌ Skipping bet - too many concurrent bets")
            return False
        
        return True
    
    def create_auto_bet(self, prediction: Dict, account: BettingAccount) -> AutoBet:
        """Create an automated bet from prediction"""
        return AutoBet(
            game_id=prediction['game_id'],
            home_team=prediction['home_team'],
            away_team=prediction['away_team'],
            predicted_outcome=prediction['predicted_outcome'],
            confidence=prediction['confidence'],
            recommended_stake=prediction['recommended_stake'],
            odds=prediction['odds'],
            sportsbook=account.platform,
            timestamp=datetime.now()
        )
    
    async def place_bet_draftkings(self, bet: AutoBet, account: BettingAccount) -> bool:
        """Place bet on DraftKings (example implementation)"""
        try:
            # DraftKings API call (example - actual API varies)
            headers = {
                'Authorization': f'Bearer {account.api_key}',
                'Content-Type': 'application/json'
            }
            
            bet_data = {
                'game_id': bet.game_id,
                'selection': bet.predicted_outcome,
                'stake': bet.recommended_stake,
                'odds': bet.odds,
                'bet_type': 'moneyline'
            }
            
            response = requests.post(
                f"{account.base_url}/api/v1/bets",
                headers=headers,
                json=bet_data,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                bet.status = "placed"
                self.logger.info(f"✅ Bet placed: {bet.home_team} vs {bet.away_team} - ${bet.recommended_stake:.2f}")
                
                # Update daily stakes
                today = datetime.now().date()
                daily_key = f"{account.platform}_{today}"
                self.daily_stakes[daily_key] = self.daily_stakes.get(daily_key, 0) + bet.recommended_stake
                
                return True
            else:
                self.logger.error(f"❌ Failed to place bet: {response.status_code} - {response.text}")
                bet.status = "failed"
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error placing bet: {e}")
            bet.status = "failed"
            return False
    
    async def place_bet_fanduel(self, bet: AutoBet, account: BettingAccount) -> bool:
        """Place bet on FanDuel (example implementation)"""
        try:
            # FanDuel API call (example)
            headers = {
                'X-API-Key': account.api_key,
                'Content-Type': 'application/json'
            }
            
            bet_data = {
                'event_id': bet.game_id,
                'market_type': 'moneyline',
                'selection': bet.predicted_outcome,
                'stake_amount': bet.recommended_stake,
                'accepted_odds': bet.odds
            }
            
            response = requests.post(
                f"{account.base_url}/v2/bets",
                headers=headers,
                json=bet_data,
                timeout=10
            )
            
            if response.status_code == 201:
                bet.status = "placed"
                self.logger.info(f"✅ FanDuel bet placed: ${bet.recommended_stake:.2f}")
                
                # Update tracking
                today = datetime.now().date()
                daily_key = f"{account.platform}_{today}"
                self.daily_stakes[daily_key] = self.daily_stakes.get(daily_key, 0) + bet.recommended_stake
                
                return True
            else:
                bet.status = "failed"
                return False
                
        except Exception as e:
            self.logger.error(f"❌ FanDuel betting error: {e}")
            bet.status = "failed"
            return False
    
    async def place_bet_crypto(self, bet: AutoBet, account: BettingAccount) -> bool:
        """Place bet on crypto sportsbook (Stake.com, etc.)"""
        try:
            # Crypto sportsbook API call
            headers = {
                'Authorization': f'Bearer {account.api_key}',
                'Content-Type': 'application/json'
            }
            
            bet_data = {
                'fixture_id': bet.game_id,
                'market': 'match_winner',
                'selection': bet.predicted_outcome,
                'amount': bet.recommended_stake,
                'currency': 'USD',  # or BTC, ETH, etc.
                'odds': bet.odds
            }
            
            response = requests.post(
                f"{account.base_url}/api/bets/place",
                headers=headers,
                json=bet_data,
                timeout=15
            )
            
            if response.status_code == 200:
                bet.status = "placed"
                self.logger.info(f"✅ Crypto bet placed: ${bet.recommended_stake:.2f}")
                return True
            else:
                bet.status = "failed"
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Crypto betting error: {e}")
            bet.status = "failed"
            return False
    
    async def process_predictions(self, predictions: List[Dict]) -> List[AutoBet]:
        """Process ML predictions and automatically place bets"""
        auto_bets = []
        
        for prediction in predictions:
            self.logger.info(f"🎯 Processing prediction: {prediction['home_team']} vs {prediction['away_team']}")
            self.logger.info(f"   Confidence: {prediction['confidence']:.1%}")
            self.logger.info(f"   Recommended stake: ${prediction['recommended_stake']:.2f}")
            
            # Try to place bet on each configured account
            for account in self.betting_accounts:
                if self.should_place_bet(prediction, account):
                    auto_bet = self.create_auto_bet(prediction, account)
                    
                    # Route to appropriate betting platform
                    success = False
                    if account.platform.lower() == 'draftkings':
                        success = await self.place_bet_draftkings(auto_bet, account)
                    elif account.platform.lower() == 'fanduel':
                        success = await self.place_bet_fanduel(auto_bet, account)
                    elif account.platform.lower() in ['stake', 'cloudbet', 'crypto']:
                        success = await self.place_bet_crypto(auto_bet, account)
                    
                    auto_bets.append(auto_bet)
                    
                    if success:
                        self.placed_bets.append(auto_bet)
                        break  # Don't place same bet on multiple platforms
        
        return auto_bets
    
    def get_daily_summary(self) -> Dict:
        """Get daily betting summary"""
        today = datetime.now().date()
        today_bets = [bet for bet in self.placed_bets if bet.timestamp.date() == today]
        
        total_staked = sum(bet.recommended_stake for bet in today_bets)
        successful_bets = len([bet for bet in today_bets if bet.status == "placed"])
        
        return {
            'date': today,
            'total_bets': len(today_bets),
            'successful_bets': successful_bets,
            'total_staked': total_staked,
            'average_confidence': sum(bet.confidence for bet in today_bets) / len(today_bets) if today_bets else 0,
            'platforms_used': list(set(bet.sportsbook for bet in today_bets))
        }

# Integration with your Bettensor system
class BettensorAutoStaker:
    """Integrates automated betting with your Bettensor MLB system"""
    
    def __init__(self, predictions_handler, betting_engine):
        self.predictions_handler = predictions_handler
        self.betting_engine = betting_engine
        self.is_running = False
        
    async def start_auto_betting(self):
        """Start the automated betting loop"""
        self.is_running = True
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("🚀 Starting automated betting system...")
        
        while self.is_running:
            try:
                # Get latest MLB predictions from your system
                mlb_predictions = self.get_mlb_predictions()
                
                if mlb_predictions:
                    self.logger.info(f"📊 Processing {len(mlb_predictions)} MLB predictions...")
                    
                    # Convert to format expected by betting engine
                    formatted_predictions = self.format_predictions(mlb_predictions)
                    
                    # Automatically place bets
                    auto_bets = await self.betting_engine.process_predictions(formatted_predictions)
                    
                    self.logger.info(f"✅ Processed {len(auto_bets)} betting opportunities")
                
                # Wait before next check (e.g., every 30 minutes)
                await asyncio.sleep(1800)
                
            except Exception as e:
                self.logger.error(f"❌ Error in auto betting loop: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    def get_mlb_predictions(self) -> List[Dict]:
        """Get MLB predictions from your Bettensor system"""
        try:
            # This would call your actual prediction system
            # predictions = self.predictions_handler.get_recent_predictions()
            
            # Example format of what your system returns
            return [
                {
                    'game_id': 'mlb_2025_001',
                    'home_team': 'New York Yankees',
                    'away_team': 'Boston Red Sox',
                    'predicted_outcome': 'New York Yankees',
                    'confidence': 0.72,
                    'recommended_stake': 45.50,
                    'odds': 1.85,
                    'start_time': '2025-07-09T19:05:00Z'
                }
            ]
        except Exception as e:
            self.logger.error(f"Error getting predictions: {e}")
            return []
    
    def format_predictions(self, predictions) -> List[Dict]:
        """Format predictions for betting engine"""
        formatted = []
        for pred in predictions:
            if hasattr(pred, 'confidence') and pred.confidence > 0.65:
                formatted.append({
                    'game_id': pred.prediction_id,
                    'home_team': pred.team_a,
                    'away_team': pred.team_b,
                    'predicted_outcome': pred.predicted_outcome,
                    'confidence': pred.confidence_score,
                    'recommended_stake': pred.wager,
                    'odds': pred.predicted_odds
                })
        return formatted

# Example usage
async def main():
    """Example of setting up automated betting"""
    
    # Configure your betting accounts
    betting_accounts = [
        BettingAccount(
            platform="DraftKings",
            api_key="your_draftkings_api_key",
            api_secret="your_draftkings_secret", 
            base_url="https://api.draftkings.com",
            balance=2000.0,
            max_daily_stake=300.0
        ),
        BettingAccount(
            platform="Stake",
            api_key="your_stake_api_key",
            api_secret="your_stake_secret",
            base_url="https://api.stake.com",
            balance=1500.0,
            max_daily_stake=200.0
        )
    ]
    
    # Create automated betting engine
    betting_engine = AutomatedBettingEngine(
        betting_accounts=betting_accounts,
        min_confidence=0.67  # Only bet on 67%+ confidence predictions
    )
    
    # Create integration with your Bettensor system
    # predictions_handler = PredictionsHandler(db_manager, state_manager)
    # auto_staker = BettensorAutoStaker(predictions_handler, betting_engine)
    
    # Start automated betting
    # await auto_staker.start_auto_betting()
    
    print("🤖 Automated betting system configured!")
    print("📊 Will automatically place bets based on ML predictions")
    print("⚡ Monitors predictions every 30 minutes")
    print("🛡️ Built-in risk management and limits")

if __name__ == "__main__":
    asyncio.run(main())
