import pandas as pd
from nba_api.stats.endpoints import playbyplayv3

game_id = '0022300061'
print(f"Đang kéo dữ liệu trận đấu {game_id} bằng PlayByPlayV3...")

# Chuyển sang dùng V3
pbp = playbyplayv3.PlayByPlayV3(game_id=game_id)
df = pbp.get_data_frames()[0]

print("\nKéo dữ liệu THÀNH CÔNG! Đây là danh sách các cột của V3:")
print(df.columns.tolist())

print("\n5 dòng dữ liệu đầu tiên:")
print(df.head())