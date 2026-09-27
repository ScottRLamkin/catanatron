import React, { useMemo, useState } from "react";
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  IconButton,
  Typography,
} from "@mui/material";
import AddIcon from "@mui/icons-material/Add";
import RemoveIcon from "@mui/icons-material/Remove";
import type { FreqDeck, PlayerState } from "../utils/api.types";
import "./DiscardSelector.scss";

const RESOURCES = ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"] as const;

type DiscardSelectorProps = {
  open: boolean;
  owed: number;
  playerState: PlayerState;
  playerKey: string;
  onDiscard: (value: FreqDeck) => void;
};

// A handful of "keep the cards for one of these" one-click presets, mirroring
// catanatron.models.discard_rule / catanrl.core.rules -- discard everything
// except what's needed for a single purchase target, in priority order.
const PRESET_COSTS: Record<string, FreqDeck> = {
  CITY: [0, 0, 0, 2, 3],
  SETTLEMENT: [1, 1, 1, 1, 0],
  DEV_CARD: [0, 0, 1, 1, 1],
  ROAD: [1, 1, 0, 0, 0],
};

function computePreset(
  hand: FreqDeck,
  owed: number,
  priority: (keyof typeof PRESET_COSTS)[]
): FreqDeck {
  const keepN = hand.reduce((a, b) => a + b, 0) - owed;
  const kept: FreqDeck = [0, 0, 0, 0, 0];
  const avail: FreqDeck = [...hand] as FreqDeck;
  let numKept = 0;
  let progress = true;
  while (numKept < keepN && progress) {
    progress = false;
    for (const target of priority) {
      const cost = PRESET_COSTS[target];
      for (let r = 0; r < 5; r++) {
        const take = Math.min(cost[r], avail[r], keepN - numKept);
        if (take > 0) {
          kept[r] += take;
          avail[r] -= take;
          numKept += take;
          progress = true;
        }
        if (numKept === keepN) break;
      }
      if (numKept === keepN) break;
    }
  }
  while (numKept < keepN) {
    let bestR = 0;
    for (let r = 1; r < 5; r++) {
      if (avail[r] > avail[bestR]) bestR = r;
    }
    kept[bestR] += 1;
    avail[bestR] -= 1;
    numKept += 1;
  }
  return RESOURCES.map((_, r) => hand[r] - kept[r]) as FreqDeck;
}

const PRESETS: { label: string; order: (keyof typeof PRESET_COSTS)[] }[] = [
  { label: "Keep for City", order: ["CITY", "SETTLEMENT", "DEV_CARD", "ROAD"] },
  {
    label: "Keep for Settlement",
    order: ["SETTLEMENT", "CITY", "DEV_CARD", "ROAD"],
  },
  { label: "Keep for Dev Card", order: ["DEV_CARD", "CITY", "SETTLEMENT", "ROAD"] },
  { label: "Keep for Road", order: ["ROAD", "SETTLEMENT", "DEV_CARD", "CITY"] },
];

export default function DiscardSelector({
  open,
  owed,
  playerState,
  playerKey,
  onDiscard,
}: DiscardSelectorProps) {
  const hand = useMemo(
    () =>
      RESOURCES.map(
        (resource) => playerState[`${playerKey}_${resource}_IN_HAND`] || 0
      ) as FreqDeck,
    [playerState, playerKey]
  );
  const [selected, setSelected] = useState<FreqDeck>([0, 0, 0, 0, 0]);

  // Reset the picker whenever it (re)opens for a new discard prompt.
  React.useEffect(() => {
    if (open) setSelected([0, 0, 0, 0, 0]);
  }, [open, owed]);

  const total = selected.reduce((a, b) => a + b, 0);
  const remaining = owed - total;

  const increment = (i: number) => {
    if (selected[i] >= hand[i] || remaining <= 0) return;
    const next = [...selected] as FreqDeck;
    next[i] += 1;
    setSelected(next);
  };
  const decrement = (i: number) => {
    if (selected[i] <= 0) return;
    const next = [...selected] as FreqDeck;
    next[i] -= 1;
    setSelected(next);
  };

  const applyPreset = (order: (keyof typeof PRESET_COSTS)[]) => {
    setSelected(computePreset(hand, owed, order));
  };

  return (
    <Dialog
      open={open}
      className="discard-selector"
      maxWidth="xs"
      fullWidth
      disableEscapeKeyDown
    >
      <DialogTitle>
        Discard {owed} card{owed === 1 ? "" : "s"}
      </DialogTitle>
      <DialogContent>
        <Typography variant="body2" className="discard-remaining">
          {remaining > 0
            ? `${remaining} more to select`
            : remaining < 0
            ? `${-remaining} too many -- remove some`
            : "Ready to discard"}
        </Typography>
        <div className="discard-grid">
          {RESOURCES.map((resource, i) => (
            <div key={resource} className={`discard-row ${resource.toLowerCase()}`}>
              <span className="resource-name">{resource}</span>
              <span className="resource-in-hand">/ {hand[i]}</span>
              <IconButton
                size="small"
                onClick={() => decrement(i)}
                disabled={selected[i] <= 0}
              >
                <RemoveIcon fontSize="small" />
              </IconButton>
              <span className="resource-count">{selected[i]}</span>
              <IconButton
                size="small"
                onClick={() => increment(i)}
                disabled={selected[i] >= hand[i] || remaining <= 0}
              >
                <AddIcon fontSize="small" />
              </IconButton>
            </div>
          ))}
        </div>
        <div className="discard-presets">
          {PRESETS.map(({ label, order }) => (
            <Button
              key={label}
              size="small"
              variant="outlined"
              onClick={() => applyPreset(order)}
            >
              {label}
            </Button>
          ))}
        </div>
      </DialogContent>
      <DialogActions>
        <Button
          variant="contained"
          color="primary"
          disabled={total !== owed}
          onClick={() => onDiscard(selected)}
        >
          Discard
        </Button>
      </DialogActions>
    </Dialog>
  );
}
