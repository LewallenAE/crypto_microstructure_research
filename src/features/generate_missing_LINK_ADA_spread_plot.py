"""
Generate missing LINK-ADA spread plot
"""
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.cointegration import plot_spread

# Generate the missing plot
print("Generating LINK-ADA spread plot...")
plot_spread('LINKUSDT', 'ADAUSDT', save_fig=True)
print("✅ Done!")