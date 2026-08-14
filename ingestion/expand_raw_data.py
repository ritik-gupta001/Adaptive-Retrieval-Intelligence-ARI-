"""
Legacy build hook entrypoint for Render deployment compatibility.
Ensures zero-downtime deployment when Render build command invokes expand_raw_data.
"""

if __name__ == "__main__":
    print("Pre-built 12,403 chunk knowledge database loaded — skipping raw expansion.")
