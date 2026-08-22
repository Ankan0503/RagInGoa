import React from "react";
import { LucideIcon } from "lucide-react";

interface CapabilityCardProps {
  key?: React.Key;
  icon: LucideIcon;
  iconStroke?: number;
  title: string;
  subtitle: string;
}

export default function CapabilityCard({
  icon: Icon,
  iconStroke = 1.8,
  title,
  subtitle,
}: CapabilityCardProps) {
  return (
    <div className="capability-card">
      <div className="capability-card-header">
        <Icon
          size={18}
          strokeWidth={iconStroke}
          className="capability-card-icon"
        />
        <span className={`capability-card-title${title.length > 8 ? " long-title" : ""}`}>
          {title}
        </span>
      </div>
      <span className="capability-card-subtitle">
        {subtitle}
      </span>
    </div>
  );
}
