---
title: IBVAP Border Surveillance
emoji: 🛰️
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: Live border CCTV analytics demo (sample clips) - SIH 2026 prototype
---

# IBVAP - Intelligent Border Video Analytics Platform

Hosted demo of a real-time CCTV analytics dashboard (YOLOv8 detection +
tracking, ANPR, face detection, virtual fence intrusion alerts, night mode
& suspicious-activity heuristics) built for SIH 2026.

**This Space runs on the bundled sample video clips only** - a hosted
container has no camera. The live-webcam version runs locally; see the
main repo for setup instructions and the full README:
https://github.com/__GITHUB_REPO__

This demo may be shared by multiple visitors at once and all see the same
live feed/state (single shared session, not per-visitor) - it's a
walkthrough demo, not a multi-tenant product.
