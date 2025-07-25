import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import SimpleConnectionPool
import warnings
from eth_utils.exceptions import ValidationError

warnings.filterwarnings("ignore", message="Network .* does not have a valid ChainId.*")
import warnings
from eth_utils.exceptions import ValidationError

warnings.filterwarnings("ignore", message="Network .* does not have a valid ChainId.*")
import bittensor as bt
import traceback
import os
import time


class DatabaseManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(DatabaseManager, cls).__new__(cls)
        return cls._instance

    def __init__(
        self,
        db_name,
        db_user,
        db_password,
        db_host="localhost",
        db_port=5432,
        max_connections=25,
    ):
        if not hasattr(self, 'initialized'):
            self.db_name = db_name
            self.db_user = db_user
            self.db_password = db_password
            self.db_host = db_host
            self.db_port = db_port
            self.max_connections = max_connections

            bt.logging.debug("Initializing DatabaseManager")
            bt.logging.debug(f"Checking root user")
            self.is_root = self.check_root_user()
            bt.logging.debug(f"Ensuring database exists")
            self.ensure_database_exists()
            bt.logging.debug(f"Waiting for database")
            self.wait_for_database()
            bt.logging.debug(f"Creating connection pool")
            self.connection_pool = self.create_connection_pool()
            bt.logging.debug(f"Creating tables")
            self.create_tables()
            bt.logging.debug("DatabaseManager initialization complete")
            self.remove_default_rows()
            self.initialized = True

    def check_root_user(self):
        return self.db_user == "root"

    def ensure_database_exists(self):
        conn = None
        try:
            # Connect to the default 'postgres' database
            conn = psycopg2.connect(
                dbname="postgres",
                user=self.db_user,
                password=self.db_password,
                host=self.db_host,
                port=self.db_port,
            )
            conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)

            with conn.cursor() as cur:
                # Check if the database exists
                cur.execute(
                    f"SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s",
                    (self.db_name,),
                )
                exists = cur.fetchone()

                if not exists:
                    bt.logging.debug(f"Creating database {self.db_name}")
                    # Create the database
                    cur.execute(f"CREATE DATABASE {self.db_name}")
                    bt.logging.debug(f"Database {self.db_name} created successfully")
                else:
                    bt.logging.debug(f"Database {self.db_name} already exists")

        except psycopg2.Error as e:
            bt.logging.error(f"Error ensuring database exists: {e}")
            raise
        finally:
            if conn:
                conn.close()

    def wait_for_database(self):
        max_retries = 5
        retry_delay = 5  # seconds

        for attempt in range(max_retries):
            conn = None
            try:
                conn = psycopg2.connect(
                    host=self.db_host,
                    port=self.db_port,
                    user=self.db_user,
                    password=self.db_password,
                    database=self.db_name,
                )
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    bt.logging.debug("Successfully connected to the database.")
                    return
            except psycopg2.OperationalError:
                if attempt < max_retries - 1:
                    bt.logging.warning(
                        f"Database not ready (attempt {attempt + 1}/{max_retries}). Retrying in {retry_delay} seconds..."
                    )
                    time.sleep(retry_delay)
                else:
                    bt.logging.error(
                        "Failed to connect to the database after multiple attempts."
                    )
                    raise
            finally:
                if conn:
                    conn.close()

    def create_connection_pool(self):
        return SimpleConnectionPool(
            1,
            self.max_connections,
            host=self.db_host,
            port=self.db_port,
            database=self.db_name,
            user=self.db_user,
            password=self.db_password,
        )

    def create_tables(self):
        tables = [
            (
                "predictions",
                """
            CREATE TABLE IF NOT EXISTS predictions (
                prediction_id TEXT PRIMARY KEY,
                game_id TEXT,
                miner_uid TEXT,
                prediction_date TEXT,
                predicted_outcome TEXT,
                predicted_odds REAL,
                team_a TEXT,
                team_b TEXT,
                wager REAL,
                team_a_odds REAL,
                team_b_odds REAL,
                tie_odds REAL,
                model_name TEXT,
                confidence_score REAL,
                outcome TEXT,
                payout REAL,
                sent_to_site INTEGER DEFAULT 0,
                validators_sent_to INTEGER DEFAULT 0,
                validators_confirmed INTEGER DEFAULT 0
            )
            """,
            ),
            (
                "games",
                """
            CREATE TABLE IF NOT EXISTS games (
                game_id TEXT PRIMARY KEY,
                team_a TEXT,
                team_b TEXT,
                sport TEXT,
                league TEXT,
                create_date TEXT,
                last_update_date TEXT,
                event_start_date TEXT,
                active INTEGER,
                outcome TEXT,
                team_a_odds REAL,
                team_b_odds REAL,
                tie_odds REAL,
                can_tie BOOLEAN
            )
            """,
            ),
            (
                "miner_stats",
                """
            CREATE TABLE IF NOT EXISTS miner_stats (
                miner_hotkey TEXT PRIMARY KEY,
                miner_coldkey TEXT,
                miner_uid TEXT,
                miner_rank INTEGER,
                miner_status TEXT,
                miner_cash REAL,
                miner_current_incentive REAL,
                miner_current_tier INTEGER,
                miner_current_scoring_window INTEGER,
                miner_current_composite_score REAL,
                miner_current_entropy_score REAL,
                miner_current_sharpe_ratio REAL,
                miner_current_sortino_ratio REAL,
                miner_current_roi REAL,
                miner_current_clv_avg REAL,
                miner_last_prediction_date TEXT,
                miner_lifetime_earnings REAL,
                miner_lifetime_wager_amount REAL,
                miner_lifetime_roi REAL,
                miner_lifetime_predictions INTEGER,
                miner_lifetime_wins INTEGER,
                miner_lifetime_losses INTEGER,
                miner_win_loss_ratio REAL,
                last_daily_reset TEXT
            )
            """,
            ),
            (
                "model_params",
                """
            CREATE TABLE IF NOT EXISTS model_params (
                miner_uid TEXT PRIMARY KEY,
                soccer_model_on BOOLEAN,
                wager_distribution_steepness INTEGER,
                fuzzy_match_percentage INTEGER,
                minimum_wager_amount FLOAT,
                max_wager_amount FLOAT,
                top_n_games INTEGER,
                nfl_model_on BOOLEAN DEFAULT FALSE,
                nfl_minimum_wager_amount FLOAT DEFAULT 20.0,
                nfl_max_wager_amount FLOAT DEFAULT 1000.0,
                nfl_top_n_games INTEGER DEFAULT 10,
                nfl_kelly_fraction_multiplier FLOAT DEFAULT 1.0,
                nfl_edge_threshold FLOAT DEFAULT 0.02,
                nfl_max_bet_percentage FLOAT DEFAULT 0.7
            );

            DO $$
            BEGIN
                IF EXISTS (
                    SELECT FROM information_schema.columns 
                    WHERE table_name = 'model_params' AND column_name = 'id'
                ) THEN
                    ALTER TABLE model_params RENAME COLUMN id TO miner_uid;
                END IF;

                IF EXISTS (
                    SELECT FROM information_schema.columns 
                    WHERE table_name = 'model_params' AND column_name = 'model_on'
                ) THEN
                    ALTER TABLE model_params RENAME COLUMN model_on TO soccer_model_on;
                END IF;
            END $$;
            """
            ),
            (
                "miner_active",
                """
            CREATE TABLE IF NOT EXISTS miner_active (
                miner_uid TEXT PRIMARY KEY,
                last_active_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            ),
            (
                "prediction_confirmations",
                """
            CREATE TABLE IF NOT EXISTS prediction_confirmations (
                prediction_id TEXT,
                validator_hotkey TEXT,
                PRIMARY KEY (prediction_id, validator_hotkey)
            )
            """,
            ),
        ]

        for table_name, create_query in tables:
            try:
                self.execute_query(create_query)
                bt.logging.debug(f"Created table: {table_name}")
            except Exception as e:
                bt.logging.error(f"Error creating table {table_name}: {e}")

    def initialize_default_model_params(self, miner_uid):
        if not miner_uid or miner_uid == "default":
            return

        bt.logging.info(f"Initializing default model params for miner: {miner_uid}")
        self.ensure_model_params_table_exists()
        self.ensure_miner_params_exist(miner_uid)

    def ensure_model_params_table_exists(self):
        query = """
        CREATE TABLE IF NOT EXISTS model_params (
            miner_uid TEXT PRIMARY KEY,
            soccer_model_on BOOLEAN,
            wager_distribution_steepness INTEGER,
            fuzzy_match_percentage INTEGER,
            minimum_wager_amount FLOAT,
            max_wager_amount FLOAT,
            top_n_games INTEGER
        );
        
        DO $$
        BEGIN
            IF EXISTS (
                SELECT FROM information_schema.columns 
                WHERE table_name = 'model_params' AND column_name = 'id'
            ) THEN
                ALTER TABLE model_params RENAME COLUMN id TO miner_uid;
            END IF;
        END $$;
        
        ALTER TABLE model_params
        ADD COLUMN IF NOT EXISTS nfl_model_on BOOLEAN DEFAULT FALSE,
        ADD COLUMN IF NOT EXISTS nfl_minimum_wager_amount FLOAT DEFAULT 20.0,
        ADD COLUMN IF NOT EXISTS nfl_max_wager_amount FLOAT DEFAULT 1000.0,
        ADD COLUMN IF NOT EXISTS nfl_top_n_games INTEGER DEFAULT 10,
        ADD COLUMN IF NOT EXISTS nfl_kelly_fraction_multiplier FLOAT DEFAULT 1.0,
        ADD COLUMN IF NOT EXISTS nfl_edge_threshold FLOAT DEFAULT 0.02,
        ADD COLUMN IF NOT EXISTS nfl_max_bet_percentage FLOAT DEFAULT 0.7,
        ADD COLUMN IF NOT EXISTS mlb_model_on BOOLEAN DEFAULT FALSE,
        ADD COLUMN IF NOT EXISTS mlb_minimum_wager_amount FLOAT DEFAULT 15.0,
        ADD COLUMN IF NOT EXISTS mlb_max_wager_amount FLOAT DEFAULT 800.0,
        ADD COLUMN IF NOT EXISTS mlb_top_n_games INTEGER DEFAULT 8,
        ADD COLUMN IF NOT EXISTS mlb_kelly_fraction_multiplier FLOAT DEFAULT 1.0,
        ADD COLUMN IF NOT EXISTS mlb_edge_threshold FLOAT DEFAULT 0.025,
        ADD COLUMN IF NOT EXISTS mlb_max_bet_percentage FLOAT DEFAULT 0.6;
        """
        self.execute_query(query)

    def ensure_miner_params_exist(self, miner_uid):
        query = """
        SELECT * FROM model_params WHERE miner_uid = %s
        """
        result = self.execute_query(query, (miner_uid,))

        if not result:
            default_params = (
                miner_uid,
                False,  # soccer_model_on
                1,      # wager_distribution_steepness
                80,     # fuzzy_match_percentage
                1.0,    # minimum_wager_amount
                100.0,  # max_wager_amount
                10,     # top_n_games
                False,  # nfl_model_on
                20.0,   # nfl_minimum_wager_amount
                1000.0, # nfl_max_wager_amount
                10,     # nfl_top_n_games
                1.0,    # nfl_kelly_fraction_multiplier
                0.02,   # nfl_edge_threshold
                0.7,    # nfl_max_bet_percentage
                False,  # mlb_model_on
                15.0,   # mlb_minimum_wager_amount
                800.0,  # mlb_max_wager_amount
                8,      # mlb_top_n_games
                1.0,    # mlb_kelly_fraction_multiplier
                0.025,  # mlb_edge_threshold
                0.6,    # mlb_max_bet_percentage
            )

            insert_query = """
            INSERT INTO model_params (
                miner_uid, soccer_model_on, wager_distribution_steepness,
                fuzzy_match_percentage, minimum_wager_amount, max_wager_amount,
                top_n_games, nfl_model_on, nfl_minimum_wager_amount,
                nfl_max_wager_amount, nfl_top_n_games, nfl_kelly_fraction_multiplier,
                nfl_edge_threshold, nfl_max_bet_percentage, mlb_model_on,
                mlb_minimum_wager_amount, mlb_max_wager_amount, mlb_top_n_games,
                mlb_kelly_fraction_multiplier, mlb_edge_threshold, mlb_max_bet_percentage
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            self.execute_query(insert_query, default_params)
            bt.logging.info(f"Created default model parameters for miner_uid: {miner_uid}")

    def get_model_params(self, miner_id):
        query = "SELECT * FROM model_params WHERE miner_uid = %s"
        result = self.execute_query(query, (miner_id,))
        return result[0] if result else None

    def update_model_params(self, miner_uid, params):
        query = """
        UPDATE model_params SET
            soccer_model_on = %s,
            wager_distribution_steepness = %s,
            fuzzy_match_percentage = %s,
            minimum_wager_amount = %s,
            max_wager_amount = %s,
            top_n_games = %s,
            nfl_model_on = %s,
            nfl_minimum_wager_amount = %s,
            nfl_max_wager_amount = %s,
            nfl_top_n_games = %s,
            nfl_kelly_fraction_multiplier = %s,
            nfl_edge_threshold = %s,
            nfl_max_bet_percentage = %s
        WHERE miner_uid = %s
        """
        self.execute_query(query, (*params.values(), miner_uid))

    def execute_query(self, query, params=None):
        conn = None
        try:
            conn = self.connection_pool.getconn()
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, params)
                if query.strip().upper().startswith("SELECT"):
                    result = cur.fetchall()
                else:
                    result = cur.rowcount
                    conn.commit()
                return result
        except Exception as e:
            bt.logging.error(f"Error executing query: {e}")
            bt.logging.error(f"Traceback: {traceback.format_exc()}")
            raise
        finally:
            if conn:
                self.connection_pool.putconn(conn)

    def execute_batch(self, query, params_list):
        conn = None
        try:
            conn = self.connection_pool.getconn()
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                bt.logging.debug(f"Executing batch query: {query}")
                bt.logging.debug(f"Number of parameter sets: {len(params_list)}")

                cur.executemany(query, params_list)
                conn.commit()
                bt.logging.debug("Batch query executed successfully")
        except Exception as e:
            if conn:
                conn.rollback()
            bt.logging.error(f"Error in execute_batch: {e}")
            bt.logging.error(f"Traceback: {traceback.format_exc()}")
            raise
        finally:
            if conn:
                self.connection_pool.putconn(conn)

    def close(self):
        self.connection_pool.closeall()

    def ensure_miner_active_table_exists(self):
        query = """
        CREATE TABLE IF NOT EXISTS miner_active (
            miner_uid TEXT PRIMARY KEY,
            last_active_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        self.execute_query(query)

    def update_miner_activity(self, miner_uid):
        query = """
        INSERT INTO miner_active (miner_uid, last_active_timestamp)
        VALUES (%s, CURRENT_TIMESTAMP AT TIME ZONE 'UTC')
        ON CONFLICT (miner_uid) DO UPDATE
        SET last_active_timestamp = CURRENT_TIMESTAMP AT TIME ZONE 'UTC'
        """
        self.execute_query(query, (miner_uid,))

    def is_miner_active(self, miner_uid):
        query = """
        SELECT COUNT(*) FROM miner_active
        WHERE miner_uid = %s AND last_active_timestamp > NOW() - INTERVAL '5 minutes'
        """
        result = self.execute_query(query, (miner_uid,))
        return result[0]["count"] > 0 if result else False

    def ensure_miner_model_params(self, miner_uid):
        query = "SELECT * FROM model_params WHERE miner_uid = %s"
        result = self.execute_query(query, (miner_uid,))

        if not result:
            default_params = {
                "soccer_model_on": False,
                "wager_distribution_steepness": 1,
                "fuzzy_match_percentage": 80,
                "minimum_wager_amount": 1.0,
                "max_wager_amount": 100.0,
                "top_n_games": 10,
                "nfl_model_on": False,
                "nfl_minimum_wager_amount": 1.0,
                "nfl_max_wager_amount": 100.0,
                "nfl_top_n_games": 5,
                "nfl_kelly_fraction_multiplier": 1.0,
                "nfl_edge_threshold": 0.02,
                "nfl_max_bet_percentage": 0.7,
            }
            insert_query = """
            INSERT INTO model_params (
                miner_uid, soccer_model_on, wager_distribution_steepness, fuzzy_match_percentage,
                minimum_wager_amount, max_wager_amount, top_n_games,
                nfl_model_on, nfl_minimum_wager_amount, nfl_max_wager_amount,
                nfl_top_n_games, nfl_kelly_fraction_multiplier, nfl_edge_threshold,
                nfl_max_bet_percentage
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            self.execute_query(insert_query, (miner_uid, *default_params.values()))
            bt.logging.info(f"Created default model parameters for miner: {miner_uid}")

    def remove_default_rows(self):
        bt.logging.info("Checking and removing default rows from all tables")
        tables = {
            "predictions": "miner_uid",
            "games": "game_id",
            "miner_stats": "miner_hotkey",
            "model_params": "miner_uid",
            "miner_active": "miner_uid",
        }

        for table, id_column in tables.items():
            query = f"""
            DELETE FROM {table}
            WHERE {id_column} = 'default' OR {id_column} IS NULL
            """
            if table == "miner_stats":
                query = f"""
                DELETE FROM {table}
                WHERE {id_column} = 'default'
                """
            try:
                rows_deleted = self.execute_query(query)
                if rows_deleted > 0:
                    bt.logging.info(
                        f"Removed {rows_deleted} default or NULL row(s) from {table}"
                    )
            except Exception as e:
                bt.logging.error(
                    f"Error removing default or NULL rows from {table}: {e}"
                )

    def get_nfl_model_status(self, miner_uid):
        query = "SELECT nfl_model_on FROM model_params WHERE miner_uid = %s"
        result = self.execute_query(query, (miner_uid,))
        return result[0]['nfl_model_on'] if result else False