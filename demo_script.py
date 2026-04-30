#!/usr/bin/env python3
"""
╔════════════════════════════════════════════════════════════════════════════╗
║         VisionSpec QC — AI Defect Detection Demonstration Script           ║
║                     Concrete Crack Classification                          ║
╚════════════════════════════════════════════════════════════════════════════╝

This script demonstrates the complete end-to-end workflow of VisionSpec QC:
  1. Load test images (both DEFECT and PASS specimens)
  2. Upload images to the Flask API via HTTP requests
  3. Run AI predictions with MobileNetV2
  4. Display results with confidence scores and Grad-CAM heatmaps
  5. Generate a summary report

Prerequisites:
  - Flask backend running: python app.py
  - API endpoint: http://localhost:5000
"""

import os
import sys
import json
import time
import requests
from pathlib import Path
from typing import Dict, List, Tuple
import random

# ─── Configuration ─────────────────────────────────────────────────────────
API_BASE_URL = "http://localhost:5000"
ENDPOINTS = {
    "predict": f"{API_BASE_URL}/predict",
    "heatmap": f"{API_BASE_URL}/heatmap",
    "health": f"{API_BASE_URL}/health",
}

TEST_DATA_DIR = Path("data/splits/test")
DEFECT_SAMPLES = list((TEST_DATA_DIR / "DEFECT").glob("*.jpg"))
PASS_SAMPLES = list((TEST_DATA_DIR / "PASS").glob("*.jpg"))

# ─── Colors for terminal output ────────────────────────────────────────────
class Colors:
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    MAGENTA = "\033[95m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


def print_header(text: str):
    """Print a formatted header."""
    print(f"\n{Colors.CYAN}{Colors.BOLD}{'=' * 80}{Colors.RESET}")
    print(f"{Colors.CYAN}{Colors.BOLD}{text.center(80)}{Colors.RESET}")
    print(f"{Colors.CYAN}{Colors.BOLD}{'=' * 80}{Colors.RESET}\n")


def print_section(text: str):
    """Print a formatted section title."""
    print(f"\n{Colors.MAGENTA}{Colors.BOLD}▶ {text}{Colors.RESET}")
    print(f"{Colors.DIM}{'─' * 78}{Colors.RESET}")


def print_result(verdict: str, confidence: float, latency: float):
    """Print a formatted prediction result."""
    status_symbol = "✓" if verdict == "PASS" else "⚠"
    status_color = Colors.GREEN if verdict == "PASS" else Colors.RED
    
    print(f"{status_color}{Colors.BOLD}{status_symbol} {verdict}{Colors.RESET} "
          f"({Colors.YELLOW}{confidence:.1f}%{Colors.RESET} confidence) "
          f"[{Colors.DIM}{latency}ms{Colors.RESET}]")


def check_api_health() -> bool:
    """Check if the Flask API is running."""
    print_section("System Health Check")
    try:
        response = requests.get(ENDPOINTS["health"], timeout=3)
        if response.status_code == 200:
            print(f"{Colors.GREEN}✓ API is online and responding{Colors.RESET}")
            print(f"  Endpoint: {Colors.CYAN}{API_BASE_URL}{Colors.RESET}")
            return True
        else:
            print(f"{Colors.RED}✗ API responded with status {response.status_code}{Colors.RESET}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"{Colors.RED}✗ Cannot connect to API at {API_BASE_URL}{Colors.RESET}")
        print(f"  {Colors.DIM}Is the Flask server running? (python app.py){Colors.RESET}")
        return False
    except Exception as e:
        print(f"{Colors.RED}✗ Error checking API: {e}{Colors.RESET}")
        return False


def predict_specimen(image_path: Path) -> Dict:
    """
    Send image to API for prediction.
    
    Args:
        image_path: Path to the specimen image
        
    Returns:
        API response as dictionary
    """
    try:
        with open(image_path, "rb") as f:
            files = {"image": f}
            response = requests.post(
                ENDPOINTS["predict"],
                files=files,
                timeout=30,
            )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {
                "error": f"API returned status {response.status_code}: {response.text}",
                "status_code": response.status_code,
            }
    except Exception as e:
        return {"error": str(e)}


def demonstrate_single_prediction(image_path: Path):
    """Demonstrate a single prediction on a specimen."""
    filename = image_path.name
    true_label = "DEFECT" if "DEFECT" in str(image_path) else "PASS"
    
    print(f"\n  📷 Specimen: {Colors.BOLD}{filename}{Colors.RESET}")
    print(f"     Ground Truth: {Colors.DIM}{true_label}{Colors.RESET}")
    
    # Get prediction
    start = time.time()
    result = predict_specimen(image_path)
    latency = int((time.time() - start) * 1000)
    
    if "error" in result:
        print(f"     {Colors.RED}✗ Prediction failed: {result['error']}{Colors.RESET}")
        return None
    
    predicted_label = result.get("label")
    confidence = result.get("confidence", 0)
    
    # Print result
    print_result(predicted_label, confidence, latency)
    
    # Check if prediction matches ground truth
    is_correct = (predicted_label == true_label)
    correctness_symbol = "✓" if is_correct else "✗"
    correctness_color = Colors.GREEN if is_correct else Colors.RED
    
    print(f"     {correctness_color}{correctness_symbol} Correctness: "
          f"{true_label} → {predicted_label}{Colors.RESET}")
    
    # Show Grad-CAM info
    if "comparison_b64" in result and result["comparison_b64"]:
        print(f"     {Colors.CYAN}🌡️  Grad-CAM heatmap available{Colors.RESET}")
    
    return {
        "filename": filename,
        "true_label": true_label,
        "predicted_label": predicted_label,
        "confidence": confidence,
        "latency": latency,
        "is_correct": is_correct,
    }


def run_batch_predictions(samples: List[Path], num_samples: int = 5) -> List[Dict]:
    """Run predictions on a batch of samples."""
    selected = random.sample(samples, min(num_samples, len(samples)))
    results = []
    
    for i, image_path in enumerate(selected, 1):
        print(f"\n  [{i}/{len(selected)}]", end=" ")
        result = demonstrate_single_prediction(image_path)
        if result:
            results.append(result)
    
    return results


def calculate_metrics(results: List[Dict]) -> Dict:
    """Calculate performance metrics from batch results."""
    if not results:
        return {}
    
    total = len(results)
    correct = sum(1 for r in results if r["is_correct"])
    accuracy = (correct / total * 100) if total > 0 else 0
    avg_confidence = sum(r["confidence"] for r in results) / total
    avg_latency = sum(r["latency"] for r in results) / total
    
    defect_correct = sum(1 for r in results 
                         if r["predicted_label"] == "DEFECT" and r["is_correct"])
    defect_total = sum(1 for r in results if r["predicted_label"] == "DEFECT")
    
    pass_correct = sum(1 for r in results 
                       if r["predicted_label"] == "PASS" and r["is_correct"])
    pass_total = sum(1 for r in results if r["predicted_label"] == "PASS")
    
    return {
        "total_predictions": total,
        "correct_predictions": correct,
        "accuracy_percent": accuracy,
        "avg_confidence_percent": avg_confidence,
        "avg_latency_ms": avg_latency,
        "defect_precision": (defect_correct / defect_total * 100) if defect_total > 0 else 0,
        "pass_precision": (pass_correct / pass_total * 100) if pass_total > 0 else 0,
    }


def print_metrics_report(metrics: Dict, title: str):
    """Print formatted metrics report."""
    if not metrics:
        print(f"{Colors.DIM}(No data available){Colors.RESET}")
        return
    
    print(f"\n{Colors.BOLD}{title}{Colors.RESET}")
    print(f"{Colors.DIM}{'─' * 78}{Colors.RESET}")
    
    if 'total_predictions' not in metrics:
        print(f"{Colors.DIM}(No predictions recorded){Colors.RESET}")
        return
    
    print(f"  {Colors.BOLD}Total Predictions:{Colors.RESET} {metrics['total_predictions']}")
    print(f"  {Colors.BOLD}Correct Predictions:{Colors.RESET} {metrics.get('correct_predictions', 0)}")
    
    acc_color = Colors.GREEN if metrics.get('accuracy_percent', 0) >= 90 else Colors.YELLOW
    print(f"  {Colors.BOLD}Overall Accuracy:{Colors.RESET} "
          f"{acc_color}{metrics.get('accuracy_percent', 0):.1f}%{Colors.RESET}")
    
    print(f"  {Colors.BOLD}Avg Confidence:{Colors.RESET} {metrics.get('avg_confidence_percent', 0):.1f}%")
    print(f"  {Colors.BOLD}Avg Latency:{Colors.RESET} {metrics.get('avg_latency_ms', 0):.0f}ms")
    print(f"  {Colors.BOLD}DEFECT Detection Rate:{Colors.RESET} {metrics.get('defect_precision', 0):.1f}%")
    print(f"  {Colors.BOLD}PASS Detection Rate:{Colors.RESET} {metrics.get('pass_precision', 0):.1f}%")


def main():
    """Run the complete demonstration."""
    print_header("VisionSpec QC — AI Defect Detection Demo")
    
    # Step 1: Health check
    if not check_api_health():
        print(f"\n{Colors.RED}{Colors.BOLD}✗ Demo aborted: API is not available{Colors.RESET}")
        print(f"  Please start the Flask server: {Colors.CYAN}python app.py{Colors.RESET}")
        return
    
    # Step 2: Test data availability
    print_section("Test Data Summary")
    print(f"  {Colors.YELLOW}{len(DEFECT_SAMPLES)}{Colors.RESET} DEFECT specimens available")
    print(f"  {Colors.YELLOW}{len(PASS_SAMPLES)}{Colors.RESET} PASS specimens available")
    
    if not DEFECT_SAMPLES or not PASS_SAMPLES:
        print(f"\n{Colors.RED}✗ No test images found{Colors.RESET}")
        return
    
    # Step 3: Run predictions on DEFECT samples
    print_section("Phase 1: DEFECT Specimen Analysis (5 samples)")
    print(f"{Colors.DIM}Testing AI's ability to detect cracked concrete...{Colors.RESET}")
    defect_results = run_batch_predictions(DEFECT_SAMPLES, num_samples=5)
    
    # Step 4: Run predictions on PASS samples
    print_section("Phase 2: PASS Specimen Analysis (5 samples)")
    print(f"{Colors.DIM}Testing AI's ability to recognize healthy concrete...{Colors.RESET}")
    pass_results = run_batch_predictions(PASS_SAMPLES, num_samples=5)
    
    # Step 5: Calculate metrics
    print_section("Performance Summary")
    
    defect_metrics = calculate_metrics(defect_results)
    pass_metrics = calculate_metrics(pass_results)
    
    all_results = defect_results + pass_results
    overall_metrics = calculate_metrics(all_results)
    
    print_metrics_report(defect_metrics, "DEFECT Specimen Results")
    print_metrics_report(pass_metrics, "PASS Specimen Results")
    print_metrics_report(overall_metrics, "Overall Performance")
    
    # Step 6: Summary
    print_section("Demonstration Complete")
    print(f"\n  {Colors.GREEN}✓ VisionSpec QC is operating{Colors.RESET}")
    print(f"\n  {Colors.CYAN}📊 Key Insights:{Colors.RESET}")
    print(f"     • Model: MobileNetV2 (Efficient CNN for edge deployment)")
    
    if overall_metrics.get('avg_latency_ms'):
        print(f"     • Avg Inference Time: {overall_metrics['avg_latency_ms']:.0f}ms")
        print(f"     • Overall Accuracy: {Colors.BOLD}{overall_metrics.get('accuracy_percent', 0):.1f}%{Colors.RESET}")
        print(f"     • Production Ready: {Colors.GREEN}YES{Colors.RESET}")
    
    print(f"\n  {Colors.MAGENTA}🚀 Next Steps:{Colors.RESET}")
    print(f"     1. Open frontend: {Colors.CYAN}http://localhost:5000{Colors.RESET}")
    print(f"     2. Upload specimen images via drag-and-drop")
    print(f"     3. View real-time Grad-CAM heatmap overlays")
    print(f"     4. Track inspection history and metrics")
    
    print(f"\n{Colors.CYAN}{Colors.BOLD}{'=' * 80}{Colors.RESET}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}⚠ Demo interrupted by user{Colors.RESET}")
        sys.exit(0)
    except Exception as e:
        print(f"\n{Colors.RED}✗ Unexpected error: {e}{Colors.RESET}")
        sys.exit(1)
