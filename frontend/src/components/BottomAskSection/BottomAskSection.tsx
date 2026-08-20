import React from "react";
import { Zap, ShieldCheck, Target, Database } from "lucide-react";
import CapabilityCard from "./CapabilityCard";
import CoastalDecorations from "./CoastalDecorations";
import "./BottomAskSection.css";

const capabilities = [
  {
    icon: Zap,
    iconStroke: 2.0,
    title: "< 200ms",
    subtitle: "End-to-end target",
  },
  {
    icon: ShieldCheck,
    iconStroke: 1.8,
    title: "Guarded",
    subtitle: "Safe & reliable",
  },
  {
    icon: Target,
    iconStroke: 1.8,
    title: "Grounded",
    subtitle: "Context-first",
  },
  {
    icon: Database,
    iconStroke: 1.8,
    title: "MSMARCO-XI",
    subtitle: "Dataset",
  },
];

export default function BottomAskSection() {
  return (
    <div className="bottom-ask-section-wrapper">
      {/* Capability Cards Grid */}
      <div className="capability-grid">
        {capabilities.map((cap, index) => (
          <CapabilityCard
            key={index}
            icon={cap.icon}
            iconStroke={cap.iconStroke}
            title={cap.title}
            subtitle={cap.subtitle}
          />
        ))}
      </div>

      {/* Coastal Decorations (Subtle waves, shell, palm leaf) */}
      <CoastalDecorations />

      {/* Signature */}
      <div className="built-in-goa">
        Built with <span className="goa-heart">❤️</span> for Goa 🌴
      </div>
    </div>
  );
}
