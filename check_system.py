"""
System Compatibility Check for AI-Powered Bank Customer Subscription Prediction System.

Run this script to verify that your environment has all required dependencies
and is ready to run the project notebook.

Usage:
    python check_system.py
"""

import sys

REQUIRED_PYTHON = (3, 8)

REQUIRED_PACKAGES = {
    "pandas": "pandas",
    "numpy": "numpy",
    "sklearn": "scikit-learn",
    "matplotlib": "matplotlib",
}


def check_python_version():
    """Check that the Python version meets the minimum requirement."""
    current = sys.version_info[:2]
    if current >= REQUIRED_PYTHON:
        print(f"  [OK] Python {current[0]}.{current[1]}")
        return True
    print(
        f"  [FAIL] Python {current[0]}.{current[1]} "
        f"(need >= {REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]})"
    )
    return False


def check_package(import_name, display_name):
    """Check that a single package can be imported and report its version."""
    try:
        mod = __import__(import_name)
        version = getattr(mod, "__version__", "unknown")
        print(f"  [OK] {display_name} ({version})")
        return True
    except ImportError:
        print(f"  [FAIL] {display_name} is not installed")
        return False


def check_sklearn_functionality():
    """Smoke-test core scikit-learn components used by the notebook."""
    try:
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder
        from sklearn.metrics import accuracy_score

        # Quick functional test with tiny synthetic data
        X = [[0, 1], [1, 0], [1, 1], [0, 0]]
        y = [0, 1, 1, 0]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.5, random_state=42
        )
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train, y_train)
        _ = accuracy_score(y_test, model.predict(X_test))

        print("  [OK] scikit-learn functionality (DecisionTreeClassifier)")
        return True
    except Exception as exc:
        print(f"  [FAIL] scikit-learn functionality: {exc}")
        return False


def main():
    print("=" * 60)
    print(" System Compatibility Check")
    print("=" * 60)

    all_ok = True

    print("\n1. Python version")
    if not check_python_version():
        all_ok = False

    print("\n2. Required packages")
    for import_name, display_name in REQUIRED_PACKAGES.items():
        if not check_package(import_name, display_name):
            all_ok = False

    print("\n3. Functionality smoke test")
    if not check_sklearn_functionality():
        all_ok = False

    print("\n" + "=" * 60)
    if all_ok:
        print(" All checks passed! Your system is ready.")
    else:
        print(" Some checks failed. Install missing packages with:")
        print("   pip install -r requirements.txt")
    print("=" * 60)

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
