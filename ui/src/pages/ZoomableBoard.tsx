import { useCallback, useContext, useEffect, useState } from "react";
import { TransformWrapper, TransformComponent } from "react-zoom-pan-pinch";
import memoize from "fast-memoize";
import { useMediaQuery, useTheme } from "@mui/material";

import useWindowSize from "../utils/useWindowSize";

import "./Board.scss";
import { store } from "../store";
import { isPlayersTurn } from "../utils/stateUtils";
import { postAction } from "../utils/apiClient";
import type { CatanState } from "../store";
import { useParams } from "react-router";
import ACTIONS from "../actions";
import Board from "./Board";
import type { GameAction, MoveRobberAction, TileCoordinate } from "../utils/api.types";
import { Button, Paper } from "@mui/material";

/**
 * Returns object representing actions to be taken if click on node.
 * @returns {3 => ["BLUE", "BUILD_CITY", 3], ...}
 */
function buildNodeActions(state: CatanState) {
  if (!state.gameState)
    throw new Error("GameState is not ready!");

  if (!isPlayersTurn(state.gameState)) {
    return {};
  }

  const nodeActions: Record<number, GameAction> = {};
  const buildInitialSettlementActions = state.gameState.is_initial_build_phase
    ? state.gameState.current_playable_actions.filter(
        (action) => action[1] === "BUILD_SETTLEMENT"
      )
    : [];
  const inInitialBuildPhase = state.gameState.is_initial_build_phase;
  if (inInitialBuildPhase) {
    buildInitialSettlementActions.forEach((action) => {
      nodeActions[action[2]] = action;
    });
  } else if (state.isBuildingSettlement) {
    state.gameState.current_playable_actions
      .filter((action) => action[1] === "BUILD_SETTLEMENT")
      .forEach((action) => {
        nodeActions[action[2]] = action;
      });
  } else if (state.isBuildingCity) {
    state.gameState.current_playable_actions
      .filter((action) => action[1] === "BUILD_CITY")
      .forEach((action) => {
        nodeActions[action[2]] = action;
      });
  }
  return nodeActions;
}

function buildEdgeActions(state: CatanState) {
  if (!state.gameState)
    throw new Error("GameState is not ready!");
  if (!isPlayersTurn(state.gameState)) {
    return {};
  }

  const edgeActions: Record<`${number},${number}`, GameAction> = {};
  const buildInitialRoadActions = state.gameState.is_initial_build_phase
    ? state.gameState.current_playable_actions.filter(
        (action) => action[1] === "BUILD_ROAD"
      )
    : [];
  const inInitialBuildPhase = state.gameState.is_initial_build_phase;
  if (inInitialBuildPhase) {
    buildInitialRoadActions.forEach((action) => {
      edgeActions[`${action[2][0]},${action[2][1]}`] = action;
      console.log(Object.keys(edgeActions), action);
    });
  } else if (state.isBuildingRoad || state.isRoadBuilding) {
    state.gameState.current_playable_actions
      .filter((action) => action[1] === "BUILD_ROAD")
      .forEach((action) => {
        edgeActions[`${action[2][0]},${action[2][1]}`] = action;
      });
  }
  return edgeActions;
}

type ZoomableBoardProps = {
  replayMode: boolean;
}

export default function ZoomableBoard({ replayMode }: ZoomableBoardProps) {
  const { gameId } = useParams();
  const { state, dispatch } = useContext(store);
  const { width, height } = useWindowSize();
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.up("md"));
  const [show, setShow] = useState(false);
  const [robberVictimChoices, setRobberVictimChoices] = useState<
    MoveRobberAction[] | null
  >(null);
  const gameState = state.gameState
  if (!gameState)
    throw new Error("GameState is not ready!");
  if (!gameId)
    throw new Error("expecting gameId in URL");

  // TODO: Move these up to GameScreen and let Zoomable be presentational component
  // https://stackoverflow.com/questions/61255053/react-usecallback-with-parameter
  const buildOnNodeClick = useCallback(
    memoize((id, action) => async () => {
      console.log("Clicked Node ", id, action);
      if (action) {
        const gameState = await postAction(gameId, action);
        dispatch({ type: ACTIONS.SET_GAME_STATE, data: gameState });
      }
    }),
    []
  );
  const buildOnEdgeClick = useCallback(
    memoize((id, action) => async () => {
      console.log("Clicked Edge ", id, action);
      if (action) {
        const gameState = await postAction(gameId, action);
        dispatch({ type: ACTIONS.SET_GAME_STATE, data: gameState });
      }
    }),
    []
  );
  const handleTileClick = useCallback(
    memoize((coordinate: TileCoordinate) => {
      console.log("Clicked Tile ", coordinate);
      if (state.isMovingRobber) {
        // Find all "MOVE_ROBBER" actions in current_playable_actions that
        // correspond to the tile coordinate selected by the user (there can
        // be more than one, one per possible victim to steal from).
        const matchingActions = gameState.current_playable_actions.filter(
          (action): action is MoveRobberAction =>
            action[1] === "MOVE_ROBBER" &&
            action[2][0].every(
              (val: number, index: number) => val === coordinate[index]
            )
        );
        if (matchingActions.length === 1) {
          postAction(gameId, matchingActions[0]).then((gameState) => {
            dispatch({ type: ACTIONS.SET_GAME_STATE, data: gameState });
          });
        } else if (matchingActions.length > 1) {
          // Multiple possible victims on this tile: let the human pick one.
          setRobberVictimChoices(matchingActions);
        }
      }
    }),
    [state.isMovingRobber]
  );
  const chooseRobberVictim = useCallback(
    (action: MoveRobberAction) => () => {
      setRobberVictimChoices(null);
      postAction(gameId, action).then((gameState) => {
        dispatch({ type: ACTIONS.SET_GAME_STATE, data: gameState });
      });
    },
    [gameId, dispatch]
  );

  const nodeActions = replayMode ? {} : buildNodeActions(state);
  const edgeActions = replayMode ? {} : buildEdgeActions(state);

  useEffect(() => {
    setTimeout(() => {
      setShow(true);
    }, 300);
  }, []);

  useEffect(() => {
    if (!state.isMovingRobber) {
      setRobberVictimChoices(null);
    }
  }, [state.isMovingRobber]);

  if (!width || !height) return;

  return (
    <TransformWrapper>
      <div className="board-container">
        <TransformComponent>
          <Board
            width={width}
            height={height}
            buildOnNodeClick={buildOnNodeClick}
            buildOnEdgeClick={buildOnEdgeClick}
            handleTileClick={handleTileClick}
            nodeActions={nodeActions}
            edgeActions={edgeActions}
            replayMode={replayMode}
            show={show}
            gameState={gameState}
            isMobile={isMobile}
            isMovingRobber={state.isMovingRobber}
          />
        </TransformComponent>
      </div>
      {robberVictimChoices && (
        <Paper
          className="robber-victim-picker"
          style={{
            position: "fixed",
            bottom: 16,
            left: "50%",
            transform: "translateX(-50%)",
            padding: 12,
            zIndex: 1300,
            display: "flex",
            flexDirection: "column",
            gap: 8,
          }}
        >
          <span>Choose who to steal from:</span>
          {robberVictimChoices.map((action) => (
            <Button
              key={action[2][1]}
              variant="contained"
              size="small"
              onClick={chooseRobberVictim(action)}
            >
              {action[2][1]}
            </Button>
          ))}
        </Paper>
      )}
    </TransformWrapper>
  );
}
