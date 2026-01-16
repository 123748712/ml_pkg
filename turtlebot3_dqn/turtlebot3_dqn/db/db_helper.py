import pymysql

DB_CONFIG = dict(
    host='192.168.0.57',
    user='dev',
    password='dev1234!',
    database='mp',
    charset="utf8"
)

class DB:
    def __init__(self, **config):
        self.config = config

    def connect(self):
        return pymysql.connect(**self.config)

    def insert_run(self):
        sql = "INSERT INTO mp.TB_RUN (START_RUN_TIME) VALUES (NOW())"
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, )
                count = cur.rowcount
                run_id = cur.lastrowid
                if count == 1 :
                    conn.commit()
                else :
                    conn.rollback()
        return run_id
    
    def insert_episode(self, episode_info):
        sql = """
            INSERT INTO mp.TB_EPISODE (EPISODE_ID, RUN_ID, START_TARGET_NUM, END_TARGET_NUM, DETECTED_COLOR, DETECTED_DIRECTION, DECISION_ACTION, SCORE, MEMORY_LENGTH, EPSILON, STEP_COUNT)
            VALUES (%s, %s, 1, 2, '', '', 0, %s, %s, %s, %s)
        """
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (episode_info['episode_id'], episode_info['run_id'], episode_info['score'], episode_info['memory_length'], episode_info['epsilon'], episode_info['step_count']))
                count = cur.rowcount
                if count == 1 :
                    conn.commit()
                else :
                    conn.rollback()
        return count == 1
        