import {SlippiGame, FramesType} from './slippi'
import * as fs from 'fs';
import * as path from 'path';

export * from './slippi'

export type Coord = {
    x: number
    y: number
}

export function processAnalogStick(coord: Coord, deadzone: boolean): Coord {
  let magnitudeSquared = (coord.x*coord.x) + (coord.y*coord.y)
  if (magnitudeSquared < 1e-3) {
    return {x: 0, y: 0}
  }

  let magnitude = Math.sqrt(magnitudeSquared)
  const threshold = 80

  let fX: number = coord.x
  let fY: number = coord.y
  if (magnitude > threshold) {
    let shrinkFactor = threshold / magnitude
    if (fX > 0) {
      fX = Math.floor(fX * shrinkFactor)
      fY = Math.floor(fY * shrinkFactor)  
    } else {
      fX = Math.ceil(fX * shrinkFactor)
      fY = Math.ceil(fY * shrinkFactor)  
    }
  }

  // Apply deadzone if applicable
  if (deadzone) {
    if (Math.abs(fX) < 23) {
      fX = 0
    }
    if (Math.abs(fY) < 23) {
      fY = 0
    }
  }

  // Round to the nearest integer (pixel)
  fX = Math.round(fX)
  fY = Math.round(fY)

  return {x: Math.floor(fX) / 80, y: Math.floor(fY) / 80}
}

export function getCoordListFromGame(game: SlippiGame, playerIndex: number, isMainStick: boolean): Coord[] {
  var frames: FramesType = game.getFrames()
  var coords: Coord[] = []
  var frame: number = -123
  while (true) {
    try {
      var coord: Coord = {x: 0, y: 0}
      var x: number = 0
      if (isMainStick) {
        x = frames[frame].players[playerIndex]?.pre.rawJoystickX
      } else {
        x = frames[frame].players[playerIndex]?.pre.cStickX
      }
      if (x !== undefined && x !== null) {
        coord.x = x
      }
      var y: number = 0
      if (isMainStick) {
        y = frames[frame].players[playerIndex]?.pre.rawJoystickY
      } else {
        y = frames[frame].players[playerIndex]?.pre.cStickY
      }
      if (y !== undefined && y !== null) {
        coord.y = y
      }

      if(isMainStick) {
        coord = processAnalogStick(coord, false)
      }

      coords.push(coord)
    }
    catch(err: any) {
      break
    } 
    frame += 1
  }
  return coords
}

export function toArrayBuffer(buffer: Buffer): ArrayBuffer {
  const arrayBuffer = new ArrayBuffer(buffer.length);
  const view = new Uint8Array(arrayBuffer);
  for (let i = 0; i < buffer.length; ++i) {
    view[i] = buffer[i];
  }
  return arrayBuffer;
}

// Function to generate JSON file from a Slippi replay
export function generateJSONFromSlippi(filePath: string, playerIndex: number, isMainStick: boolean) {
    const slpDir = path.join(filePath)
    let data = fs.readFileSync(slpDir, null);
    let game = new SlippiGame(toArrayBuffer(data));
    const coordinates = getCoordListFromGame(game, playerIndex, isMainStick);

    // Convert coordinates to JSON and save to file
    fs.writeFileSync(path.join(outputPath, `coordinates${playerIndex}.json`), JSON.stringify(coordinates));
}
  
// Get file path from command-line arguments
const args = process.argv.slice(2);
if (args.length < 2) {
    console.error("Usage: ts-node generate_coordinates.ts <slippi_replay_file> <output_path>");
    process.exit(1);
}

const filePath = args[0];
const outputPath = args[1];
const isMainStick = true;

for (let i = 0; i < 4; ++i) {
  generateJSONFromSlippi(filePath, i, isMainStick);
}
