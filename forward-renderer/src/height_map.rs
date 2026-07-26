//! Height map for fast height location lookups for the physics engine

pub mod height_map_drawer;

type Vec3 = cgmath::Vector3<f32>;


pub struct HeightMap<
    const WIDTH: usize,
    const HEIGHT: usize,
    const TILE_WIDTH: usize,
    const TILE_HEIGHT: usize,
> {
    pub data: Vec<[[f32; TILE_WIDTH]; TILE_HEIGHT]>,
}

impl<const WIDTH: usize, const HEIGHT: usize, const TILE_WIDTH: usize, const TILE_HEIGHT: usize> Default for HeightMap<WIDTH, HEIGHT, TILE_WIDTH, TILE_HEIGHT> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const WIDTH: usize, const HEIGHT: usize, const TILE_WIDTH: usize, const TILE_HEIGHT: usize>
    HeightMap<WIDTH, HEIGHT, TILE_WIDTH, TILE_HEIGHT>
{
    const INNER_WIDTH: usize = TILE_WIDTH - 2;
    const INNER_HEIGHT: usize = TILE_HEIGHT - 2;
    const INNER_SIZE: usize = Self::INNER_WIDTH * Self::INNER_HEIGHT;

    const TILES_X: usize = WIDTH / Self::INNER_WIDTH;
    const TILES_Y: usize = HEIGHT / Self::INNER_HEIGHT;
    const TILE_COUNT: usize = Self::TILES_X * Self::TILES_Y;

    const OFFSET_X: usize = Self::TILES_X / 2;
    const OFFSET_Y: usize = Self::TILES_X / 2;

    pub fn new() -> Self {
        let mut data = Vec::with_capacity(Self::TILE_COUNT);
        for _i in 0..Self::TILE_COUNT {
            data.push([[0.0; TILE_WIDTH]; TILE_HEIGHT]);
        }

        Self { data }
    }

    pub fn set_tile(&mut self, tile_y: usize, tile_x: usize, data: &[f32]){
        assert_eq!(data.len(), Self::INNER_SIZE);

        // get tile
        let tile = &mut self.data[tile_y * Self::TILES_X + tile_x];

        // write inner data
        for y in 0..Self::INNER_HEIGHT {
            for x in 0..Self::INNER_WIDTH {
                let index = y * Self::INNER_WIDTH + x;
                tile[y+1][x+1] = data[index];
            }
        }

        // Copy top edge to top neighbor
        if tile_y > 0 {
            let top = &mut self.data[(tile_y - 1) * Self::TILES_X + tile_x];

            for x in 0..Self::INNER_WIDTH {
                let index = x;
                top[TILE_HEIGHT - 1][x + 1] = data[index];
            }
        }

        // Copy bottom edge to bottom neighbor
        if tile_y < Self::TILES_Y - 1 {
            let bottom = &mut self.data[(tile_y + 1) * Self::TILES_X + tile_x];

            for x in 0..Self::INNER_WIDTH {
                let index = (Self::INNER_HEIGHT - 1) * Self::INNER_WIDTH + x;
                bottom[0][x + 1] = data[index];
            }
        }

        // Copy left edge to left neighbor
        if tile_x > 0 {
            let left = &mut self.data[tile_y * Self::TILES_X + (tile_x - 1)];

            for y in 0..Self::INNER_HEIGHT {
                let index = y * Self::INNER_WIDTH;
                left[y + 1][TILE_WIDTH - 1] = data[index];
            }
        }

        // Copy right edge to right neighbor
        if tile_x < Self::TILES_X - 1 {
            let right = &mut self.data[tile_y * Self::TILES_X + (tile_x + 1)];

            for y in 0..Self::INNER_HEIGHT {
                let index = y * Self::INNER_WIDTH + (Self::INNER_WIDTH - 1);
                right[y + 1][0] = data[index];
            }
        }

        // Copy top-left corner
        if tile_x > 0 && tile_y > 0 {
            let top_left = &mut self.data[(tile_y - 1) * Self::TILES_X + tile_x - 1];

            top_left[TILE_HEIGHT - 1][TILE_WIDTH - 1] = data[0];
        }

        // Copy top-right corner
        if tile_x < Self::TILES_X - 1 && tile_y > 0 {
            let top_right = &mut self.data[(tile_y - 1) * Self::TILES_X + tile_x + 1];

            top_right[TILE_HEIGHT - 1][0] = data[Self::INNER_WIDTH - 1];
        }

        // Copy bottom-left corner
        if tile_x > 0 && tile_y < Self::TILES_Y - 1 {
            let bottom_left = &mut self.data[(tile_y + 1) * Self::TILES_X + tile_x - 1];

            bottom_left[0][TILE_WIDTH - 1] =
                data[(Self::INNER_HEIGHT - 1) * Self::INNER_WIDTH];
        }

        // Copy bottom-right corner
        if tile_x < Self::TILES_X - 1 && tile_y < Self::TILES_Y - 1 {
            let bottom_right = &mut self.data[(tile_y + 1) * Self::TILES_X + tile_x + 1];

            bottom_right[0][0] =
                data[(Self::INNER_HEIGHT - 1) * Self::INNER_WIDTH + Self::INNER_WIDTH - 1];
        }
    }

    pub fn get_height(&self, y: f32, x: f32) -> f32 {
        // move to the middle
        let y = y + (Self::OFFSET_Y * Self::INNER_HEIGHT) as f32;
        let x = x + (Self::OFFSET_X * Self::INNER_WIDTH) as f32;

        // Clamp to valid world coordinates
        let x = x.clamp(0.0, (WIDTH - 1) as f32);
        let y = y.clamp(0.0, (HEIGHT - 1) as f32);

        // Find tile
        let tile_x = ((x as usize) / Self::INNER_WIDTH).min(Self::TILES_X - 1);
        let tile_y = ((y as usize) / Self::INNER_HEIGHT).min(Self::TILES_Y - 1);

        let tile = &self.data[tile_y * Self::TILES_X + tile_x];

        // Position inside tile's inner area
        // +1 accounts for the overlapping border
        let local_x = x - (tile_x * Self::INNER_WIDTH) as f32 + 1.0;
        let local_y = y - (tile_y * Self::INNER_HEIGHT) as f32 + 1.0;

        let x0 = local_x.floor() as usize;
        let y0 = local_y.floor() as usize;

        let x1 = (x0 + 1).min(TILE_WIDTH - 1);
        let y1 = (y0 + 1).min(TILE_HEIGHT - 1);

        let tx = local_x - x0 as f32;
        let ty = local_y - y0 as f32;

        let h00 = tile[y0][x0];
        let h10 = tile[y0][x1];
        let h01 = tile[y1][x0];
        let h11 = tile[y1][x1];

        // Bilinear interpolation
        let hx0 = h00 * (1.0 - tx) + h10 * tx;
        let hx1 = h01 * (1.0 - tx) + h11 * tx;

        hx0 * (1.0 - ty) + hx1 * ty
    }
}

pub trait HeightMapInterface {
    fn get_height(&self, pos: &Vec3) -> f32;
}

impl<const WIDTH: usize, const HEIGHT: usize, const TILE_WIDTH: usize, const TILE_HEIGHT: usize>
HeightMapInterface for 
    HeightMap<WIDTH, HEIGHT, TILE_WIDTH, TILE_HEIGHT>
{
    fn get_height(&self, pos: &Vec3) -> f32 {
        let height = self.get_height(pos.y, pos.x);

        height
    }
}