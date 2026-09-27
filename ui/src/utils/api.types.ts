export type Card = ResourceCard | DevelopmentCard | VictoryPointCard;
export type DevelopmentCard =
  | "KNIGHT"
  | "MONOPOLY"
  | "YEAR_OF_PLENTY"
  | "ROAD_BUILDING";

export type Color = "RED" | "BLUE" | "ORANGE" | "WHITE";
export type TileCoordinate = [number, number, number];

export type GameActionRecord =
  // These are the special cases
  | [RollGameAction, [number, number]]
  | [DiscardGameAction, null]
  | [MoveRobberAction, ResourceCard | null]
  | [BuyDevelopmentCardAction, DevelopmentCard]
  // These are deterministic and carry no extra info
  | [BuildSettlementAction, null]
  | [BuildCityAction, null]
  | [BuildRoadAction, null]
  | [PlayKnightCardAction, null]
  | [PlayRoadBuildingAction, null]
  | [PlayMonopolyAction, null]
  | [PlayYearOfPlentyAction, null]
  | [MaritimeTradeAction, null]
  | [OfferTradeAction, null]
  | [AcceptTradeAction, null]
  | [RejectTradeAction, null]
  | [ConfirmTradeAction, null]
  | [CancelTradeAction, null]
  | [EndTurnAction, null];

export type RollGameAction = [Color, "ROLL", null];
// A discard is one whole set, chosen at once: a 5-tuple freqdeck (WOOD,
// BRICK, SHEEP, WHEAT, ORE) summing to what the player owed. Submissions
// during a still-open 7-roll round (state.is_discarding) are hidden from
// everyone (including the discarder's own action log entry) until every
// owing player has submitted, at which point the engine reveals the real
// values retroactively -- so `value` is `null` for a still-pending entry.
export type FreqDeck = [number, number, number, number, number];
export type DiscardGameAction = [Color, "DISCARD", FreqDeck | null];
export type BuyDevelopmentCardAction = [Color, "BUY_DEVELOPMENT_CARD", null];
export type BuildSettlementAction = [Color, "BUILD_SETTLEMENT", number];
export type BuildCityAction = [Color, "BUILD_CITY", number];
export type BuildRoadAction = [Color, "BUILD_ROAD", [number, number]];
export type PlayKnightCardAction = [Color, "PLAY_KNIGHT_CARD", null];
export type PlayRoadBuildingAction = [Color, "PLAY_ROAD_BUILDING", null];
export type PlayMonopolyAction = [Color, "PLAY_MONOPOLY", ResourceCard];
export type PlayYearOfPlentyAction = [
  Color,
  "PLAY_YEAR_OF_PLENTY",
  [ResourceCard] | [ResourceCard, ResourceCard]
];
export type MoveRobberAction = [
  Color,
  "MOVE_ROBBER",
  [TileCoordinate, Color | null]
];
export type MaritimeTradeAction = [
  Color,
  "MARITIME_TRADE",
  (ResourceCard | null)[]
];

// Domestic (player-to-player) trade. `TradeOffer` is a 10-tuple: 5 offered
// resource counts (freqdeck order WOOD, BRICK, SHEEP, WHEAT, ORE), followed
// by 5 asked resource counts.
export type TradeOffer = [
  number,
  number,
  number,
  number,
  number,
  number,
  number,
  number,
  number,
  number
];
// state.current_trade: TradeOffer (10) + offerer's seat index (1) = 11-tuple.
export type CurrentTrade = [...TradeOffer, number];
export type OfferTradeAction = [Color, "OFFER_TRADE", TradeOffer];
export type AcceptTradeAction = [Color, "ACCEPT_TRADE", CurrentTrade];
export type RejectTradeAction = [Color, "REJECT_TRADE", CurrentTrade];
export type ConfirmTradeAction = [
  Color,
  "CONFIRM_TRADE",
  [...TradeOffer, Color]
];
export type CancelTradeAction = [Color, "CANCEL_TRADE", null];

export type EndTurnAction = [Color, "END_TURN", null];

export type GameAction =
  | RollGameAction
  | DiscardGameAction
  | BuyDevelopmentCardAction
  | BuildSettlementAction
  | BuildCityAction
  | BuildRoadAction
  | PlayKnightCardAction
  | PlayRoadBuildingAction
  | PlayMonopolyAction
  | PlayYearOfPlentyAction
  | MoveRobberAction
  | MaritimeTradeAction
  | OfferTradeAction
  | AcceptTradeAction
  | RejectTradeAction
  | ConfirmTradeAction
  | CancelTradeAction
  | EndTurnAction;

export type PlayerState = any;
export type VictoryPointCard = "VICTORY_POINT";
export type ResourceCard = "WOOD" | "BRICK" | "SHEEP" | "WHEAT" | "ORE";
export type Building = "SETTLEMENT" | "CITY";

type ResourceTile = {
  id: number;
  type: "RESOURCE_TILE";
  resource: ResourceCard;
  number: number;
};

type DesertTile = {
  id: number;
  type: "DESERT";
};

type PortTile = {
  id: number;
  type: "PORT";
  direction: Direction;
  resource: ResourceCard;
};

export type Tile = ResourceTile | DesertTile | PortTile;

export type PlacedTile = {
  coordinate: TileCoordinate;
  tile: Tile;
};

export type GameState = {
  tiles: PlacedTile[];
  adjacent_tiles: Record<string, Tile[]>;
  bot_colors: Color[];
  colors: Color[];
  current_color: Color;
  winning_color?: Color;
  current_prompt: string;
  player_state: Record<string, PlayerState>;
  action_records: GameActionRecord[];
  robber_coordinate: TileCoordinate;
  nodes: Array<{
    id: number;
    tile_coordinate: TileCoordinate;
    direction: Direction;
    building: Building | null;
    color: Color | null;
  }>;
  edges: Array<{
    id: [number, number];
    color: Color | null;
    direction: Direction;
    tile_coordinate: TileCoordinate;
  }>;
  current_playable_actions: GameAction[];
  is_initial_build_phase: boolean;
  edgeActions?: GameAction[];
  nodeActions?: GameAction[];
  state_index: number;
  current_trade: CurrentTrade;
  acceptees: Record<Color, boolean>;
  turn_trade_offers: TradeOffer[];
  max_trade_offers_per_turn: number;
};
const DIRECTIONS = [
  "NORTH",
  "NORTHEAST",
  "SOUTHEAST",
  "SOUTH",
  "SOUTHWEST",
  "NORTHWEST",
  "EAST",
  "WEST",
] as const;

export type Direction = (typeof DIRECTIONS)[number];
