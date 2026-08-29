//! A universal function approximator

pub mod epoch;

#[cfg(test)]
mod tests;

type Vec2 = nalgebra::Vector2<f32>;
type Mat2 = nalgebra::Matrix2<f32>;
type RowVec2 = nalgebra::RowVector2<f32>;
type RowVec4 = nalgebra::RowVector4<f32>;

#[derive(Debug, Clone)]
pub struct NeuralNetwork {

    // forward

    pub x: Vec2,

    pub w_0: Mat2,
    pub w_1: Mat2,
    pub w_2: Mat2,
    pub w_3: RowVec2,

    pub b_0: Vec2,
    pub b_1: Vec2,
    pub b_2: Vec2,
    pub b_3: f32,

    z_0: Vec2,
    z_1: Vec2,
    z_2: Vec2,
    z_3: f32,

    a_0: Vec2,
    a_1: Vec2,
    a_2: Vec2,

    pub y: f32,

    // backward

    pub dy_dw0: RowVec4,
    dy_dw1: RowVec4,
    dy_dw2: RowVec4,
    dy_dw3: RowVec2,

    pub dy_db0: RowVec2,
    dy_db1: RowVec2,
    dy_db2: RowVec2,
    dy_db3: f32,
}

impl NeuralNetwork {
    pub fn new() -> Self {

        // forward

        let x = Vec2::new(1.0, 1.0);

        let w_0 = Mat2::new(1.0, 1.0, 1.0, 1.0);
        let w_1 = Mat2::new(1.0, 1.0, 1.0, 1.0);
        let w_2 = Mat2::new(1.0, 1.0, 1.0, 1.0);
        let w_3 = RowVec2::new(1.0, 1.0);

        let b_0 = Vec2::new(1.0, 1.0);
        let b_1 = Vec2::new(1.0, 1.0);
        let b_2 = Vec2::new(1.0, 1.0);
        let b_3 = 1.0;

        let z_0 = Vec2::new(0.0, 0.0);
        let z_1 = Vec2::new(0.0, 0.0);
        let z_2 = Vec2::new(0.0, 0.0);
        let z_3 = 0.0;

        let a_0 = Vec2::new(0.0, 0.0);
        let a_1 = Vec2::new(0.0, 0.0);
        let a_2 = Vec2::new(0.0, 0.0);

        let y = 0.0;

        // backward

        let dy_dw0 = RowVec4::new(0.0, 0.0, 0.0, 0.0);
        let dy_dw1 = RowVec4::new(0.0, 0.0, 0.0, 0.0);
        let dy_dw2 = RowVec4::new(0.0, 0.0, 0.0, 0.0);
        let dy_dw3 = RowVec2::new(0.0, 0.0);

        let dy_db0 = RowVec2::new(0.0, 0.0);
        let dy_db1 = RowVec2::new(0.0, 0.0);
        let dy_db2 = RowVec2::new(0.0, 0.0);
        let dy_db3 = 0.0;

        Self {

            // forward

            x,

            w_0,
            w_1,
            w_2,
            w_3,

            b_0,
            b_1,
            b_2,
            b_3,
            
            z_0,
            z_1,
            z_2,
            z_3,

            a_0,
            a_1,
            a_2,

            y,

            // backward

            dy_dw0,
            dy_dw1,
            dy_dw2,
            dy_dw3,

            dy_db0,
            dy_db1,
            dy_db2,
            dy_db3,
        }
    }

    pub fn forward(&mut self) 
    {
        self.z_0 = self.w_0 * self.x + self.b_0;
        self.a_0 = Self::_activation_re_lu_vec2(self.z_0);

        self.z_1 = self.w_1 * self.a_0 + self.b_1;
        self.a_1 = Self::_activation_re_lu_vec2(self.z_1);

        self.z_2 = self.w_2 * self.a_1 + self.b_2;
        self.a_2 = Self::_activation_re_lu_vec2(self.z_2);

        self.y = (self.w_3 * self.a_2)[(0, 0)] + self.b_3;
    }

    pub fn backward(&mut self) 
    {
        // weight gradients

        self.dy_dw3 = Self::to_1x2(self.a_2);

        self.dy_dw2 = self.w_3 * Self::_derivative_re_lu_vec2(self.z_2) *
                    Self::to_2x4(self.a_1);

        self.dy_dw1 = self.w_3 * Self::_derivative_re_lu_vec2(self.z_2) *
                    self.w_2 * Self::_derivative_re_lu_vec2(self.z_1) *
                    Self::to_2x4(self.a_0);

        self.dy_dw0 = self.w_3 * Self::_derivative_re_lu_vec2(self.z_2) *
                    self.w_2 * Self::_derivative_re_lu_vec2(self.z_1) *
                    self.w_1 * Self::_derivative_re_lu_vec2(self.z_0) *
                    Self::to_2x4(self.x);

        // bias gradients

        self.dy_db3 = 1.0;

        self.dy_db2 = self.w_3 * Self::_derivative_re_lu_vec2(self.z_2);

        self.dy_db1 = self.w_3 * Self::_derivative_re_lu_vec2(self.z_2) *
                    self.w_2 * Self::_derivative_re_lu_vec2(self.z_1);

        self.dy_db0 = self.w_3 * Self::_derivative_re_lu_vec2(self.z_2) *
                    self.w_2 * Self::_derivative_re_lu_vec2(self.z_1) *
                    self.w_1 * Self::_derivative_re_lu_vec2(self.z_0);
    }

    fn to_2x4(val: Vec2) -> nalgebra::Matrix2x4<f32> {
        nalgebra::Matrix2x4::new(
            val.x, val.y, 0.0, 0.0,
            0.0, 0.0, val.x, val.y,
        )
    }

    fn to_1x2(val: Vec2) -> RowVec2 {
        val.transpose()
    }

    // activation re_lu

    fn _activation_re_lu_vec2(value: Vec2) -> Vec2 {
       Vec2::new(
            Self::_activation_re_lu(value.x), 
            Self::_activation_re_lu(value.y), 
        )
    }

    fn _derivative_re_lu_vec2(value: Vec2) -> Mat2 {
        Mat2::new(
            Self::_derivative_re_lu(value.x),
            0.0,
            0.0,
            Self::_derivative_re_lu(value.y)
        )
    }

    fn _activation_re_lu(value: f32) -> f32 {
        value.max(0.0)

        // // leaky relu
        // if value > 0.0 {
        //     value
        // } else {
        //     0.01 * value
        // }
    }

    fn _derivative_re_lu(value: f32) -> f32 {
        if value > 0.0 {
            1.0
        } else {
            0.0
        }

        // // leaky relu
        // if value > 0.0 {
        //     1.0
        // } else {
        //     0.01
        // }
    }
}


impl std::fmt::Display for NeuralNetwork {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "NeuralNetwork {{")?;
        
        writeln!(f, "x: {:?}", self.x)?;
        writeln!(f, )?;

        writeln!(f, "w_0: {:?}", self.w_0)?;
        writeln!(f, "w_1: {:?}", self.w_1)?;
        writeln!(f, "w_2: {:?}", self.w_2)?;
        writeln!(f, "w_3: {:?}", self.w_3)?;
        writeln!(f, )?;

        writeln!(f, "b_0: {:?}", self.b_0)?;
        writeln!(f, "b_1: {:?}", self.b_1)?;
        writeln!(f, "b_2: {:?}", self.b_2)?;
        writeln!(f, "b_3: {:?}", self.b_3)?;
        writeln!(f, )?;

        writeln!(f, "z_0: {:?}", self.z_0)?;
        writeln!(f, "z_1: {:?}", self.z_1)?;
        writeln!(f, "z_2: {:?}", self.z_2)?;
        writeln!(f, "z_3: {:?}", self.z_3)?;
        writeln!(f, )?;

        writeln!(f, "a_0: {:?}", self.a_0)?;
        writeln!(f, "a_1: {:?}", self.a_1)?;
        writeln!(f, "a_2: {:?}", self.a_2)?;
        writeln!(f, )?;

        writeln!(f, "y: {:?}", self.y)?;
        writeln!(f, )?;

        writeln!(f, "dw_0: {:?}", self.dy_dw0)?;
        writeln!(f, "dw_1: {:?}", self.dy_dw1)?;
        writeln!(f, "dw_2: {:?}", self.dy_dw2)?;
        writeln!(f, "dw_3: {:?}", self.dy_dw3)?;
        writeln!(f, )?;

        writeln!(f, "db_0: {:?}", self.dy_db0)?;
        writeln!(f, "db_1: {:?}", self.dy_db1)?;
        writeln!(f, "db_2: {:?}", self.dy_db2)?;
        writeln!(f, "db_3: {:?}", self.dy_db3)?;
        writeln!(f, )?;

        write!(f, "}}")
    }
}
