//! A universal function approximator

#[cfg(test)]
mod tests;

type Vec2 = nalgebra::Vector2<f32>;
type Mat2 = nalgebra::Matrix2<f32>;
type RowVec2 = nalgebra::RowVector2<f32>;
type RowVec4 = nalgebra::RowVector4<f32>;

#[derive(Debug, Clone)]
struct NeuralNetwork {

    // forward

    x: Vec2,

    w_0: Mat2,
    w_1: Mat2,
    w_2: Mat2,
    w_3: RowVec2,

    b_0: Vec2,
    b_1: Vec2,
    b_2: Vec2,
    b_3: f32,

    z_0: Vec2,
    z_1: Vec2,
    z_2: Vec2,
    z_3: f32,

    a_0: Vec2,
    a_1: Vec2,
    a_2: Vec2,

    y: f32,

    // backward

    dw_0: RowVec4,
    dw_1: RowVec4,
    dw_2: RowVec4,
    dw_3: RowVec2,

}

impl NeuralNetwork {
    fn new() -> Self {

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

        let dw_0 = RowVec4::new(0.0, 0.0, 0.0, 0.0);
        let dw_1 = RowVec4::new(0.0, 0.0, 0.0, 0.0);
        let dw_2 = RowVec4::new(0.0, 0.0, 0.0, 0.0);
        let dw_3 = RowVec2::new(0.0, 0.0);

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

            dw_0,
            dw_1,
            dw_2,
            dw_3,
        }
    }

    fn forward(&mut self) 
    {
        self.z_0 = self.w_0 * self.x + self.b_0;
        self.a_0 = Self::_activation_re_lu_vec2(self.z_0);

        self.z_1 = self.w_1 * self.a_0 + self.b_1;
        self.a_1 = Self::_activation_re_lu_vec2(self.z_1);

        self.z_2 = self.w_2 * self.a_1 + self.b_2;
        self.a_2 = Self::_activation_re_lu_vec2(self.z_2);

        self.z_3 = (self.w_3 * self.a_2)[(0, 0)] + self.b_3;
        self.y = Self::_activation_re_lu(self.z_3); 

        // todo: Loss function
    }

    fn backward(&mut self) 
    {
        self.dw_3 = Self::_derivative_re_lu(self.z_3) * 
                    Self::to_1x2(self.a_2);

        self.dw_2 = Self::_derivative_re_lu(self.z_3) * 
                    self.w_3 * Self::_derivative_re_lu_vec2(self.z_2) *
                    Self::to_2x4(self.a_1);

        self.dw_1 = Self::_derivative_re_lu(self.z_3) * 
                    self.w_3 * Self::_derivative_re_lu_vec2(self.z_2) *
                    self.w_2 * Self::_derivative_re_lu_vec2(self.z_1) *
                    Self::to_2x4(self.a_0);

        self.dw_0 = Self::_derivative_re_lu(self.z_3) * 
                    self.w_3 * Self::_derivative_re_lu_vec2(self.z_2) *
                    self.w_2 * Self::_derivative_re_lu_vec2(self.z_1) *
                    self.w_1 * Self::_derivative_re_lu_vec2(self.z_0) *
                    Self::to_2x4(self.x);

        // todo: bias gradients

        // todo: Loss function
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
    }

    fn _derivative_re_lu(value: f32) -> f32 {
        if value > 0.0 {
            1.0
        } else {
            0.0
        }
    }
}


impl std::fmt::Display for NeuralNetwork {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "NeuralNetwork {{")?;
        
        println!("x: {:?}", self.x);
        println!("");

        println!("w_0: {:?}", self.w_0);
        println!("w_1: {:?}", self.w_1);
        println!("w_2: {:?}", self.w_2);
        println!("w_3: {:?}", self.w_3);
        println!("");

        println!("b_0: {:?}", self.b_0);
        println!("b_1: {:?}", self.b_1);
        println!("b_2: {:?}", self.b_2);
        println!("b_3: {:?}", self.b_3);
        println!("");

        println!("z_0: {:?}", self.z_0);
        println!("z_1: {:?}", self.z_1);
        println!("z_2: {:?}", self.z_2);
        println!("z_3: {:?}", self.z_3);
        println!("");

        println!("a_0: {:?}", self.a_0);
        println!("a_1: {:?}", self.a_1);
        println!("a_2: {:?}", self.a_2);
        println!("");

        println!("y: {:?}", self.y);
        println!("");

        println!("dw_0: {:?}", self.dw_0);
        println!("dw_1: {:?}", self.dw_1);
        println!("dw_2: {:?}", self.dw_2);
        println!("dw_3: {:?}", self.dw_3);
        println!("");

        write!(f, "}}")
    }
}


    // activation x²

    // fn _activation_x2_vec2(value: Vec2) -> Vec2 {
    //    Vec2::new(
    //         Self::_activation_x2(value.x), 
    //         Self::_activation_x2(value.y), 
    //     )
    // }

    // fn _derivative_x2_vec2(value: Vec2) -> Mat2 {
    //     Mat2::new(
    //         Self::_derivative_x2(value.x),
    //         0.0,
    //         0.0,
    //         Self::_derivative_x2(value.y)
    //     )
    // }

    // fn _activation_x2(value: f32) -> f32 {
    //     value * value
    // }

    // fn _derivative_x2(value: f32) -> f32 {
    //     2.0 * value
    // }
