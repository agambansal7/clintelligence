from sqlalchemy import create_engine, text
import os
engine = create_engine(os.environ['DATABASE_URL']) 
with engine.connect() as conn: 
conn.execute(text('DROP TABLE IF EXISTS trials CASCADE'))
conn.execute(text('DROP TABLE IF EXISTS sites CASCADE'))
conn.execute(text('DROP TABLE IF EXISTS endpoints CASCADE'))
conn.execute(text('DROP TABLE IF EXISTS investigators CASCADE'))
conn.execute(text('DROP TABLE IF EXISTS trial_benchmarks CASCADE')) 
conn.commit()
print('Tables cleared!')
