# Galaxea A1 Collection Console

Private Foxglove organization extension for guarded A1 collection controls.
It subscribes only to the sanitized workflow-status topic and calls the five
exact ROS `std_srvs/Trigger` services exposed by the A1 telemetry adapter. Its
English-only UI contains one status and five minimal, monochrome controls.

Run `just foxglove-layout` from the repository root after changing System topic
or service names. Build locally with `npm ci && npm run build`; pushes to `main`
package and publish the extension before updating the canonical organization
layout.
