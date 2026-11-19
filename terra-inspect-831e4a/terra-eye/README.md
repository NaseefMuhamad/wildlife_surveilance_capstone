# TERRA-EYE // Wildlife Telemetry

A responsive React + Tailwind + Lucide prototype of a wildlife surveillance and conservation dashboard.

## Run locally

```bash
npm install
npm run dev
```

Open the URL Vite prints, usually http://localhost:5173.

## Features

- Collapsible camera-trap sidebar grouped by sectors, with battery, storage and link status.
- 1-camera / 4-camera monitoring layouts, camera selection, and simulated IR/thermal overlays.
- SVG topographic map with sector camera positions, selection, and illustrative animal movement trails.
- Live-look event feed with species and security filters, search, confidence scores, and simulated event injection.
- Notifications, sidebar tabs, responsive layout, and data export as CSV.
- Data is **entirely mock**, including feeds, timestamps, geolocation and detection confidence. No real cameras, wildlife locations, AI models, GPS devices or weather services are connected.

## Production integrations

Replace mock data with authenticated camera/NVR endpoints (e.g. RTSP via server-side WebRTC/HLS gateway), telemetry ingestion (MQTT/HTTPS), geospatial backend, and a verified event detection service. Restrict sensitive wildlife-location access and apply retention policies before deployment.
