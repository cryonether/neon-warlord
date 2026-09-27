//! Automatic Differentiation Engine


enum Op {
    Add {
        lhs: f32,
        rhs: f32,
    },

    Mul {
        lhs: f32,
        rhs: f32,
    },

    Exp {
        input: f32,
    },

    Log {
        input: f32,
    }
}

struct Value {
    data: f32,
    grad: f32,
    op: Op,
}

struct AutoDiff {
    x: f32,
    res: f32,

    graph: Vec<Op>,
}

impl AutoDiff {
    fn new(x: f32) -> Self {

        let res = x;
        let graph = Vec::new();

        Self {
            x,
            res,
            graph,
        }
    }

    pub fn mul(mut self, rhs: f32) -> Self {
        
        self.graph.push(Op::Mul{lhs: self.res, rhs});

        self.res *= rhs;

        self
    }

    pub fn backward(&mut self) -> f32 {
        0.0
    }
}



#[test]
fn test_mul() {
    

    let x = 2.0;
    let a = 3.0;

    let y = x * a; 
    assert_eq!(y, 6.0);


    let x = AutoDiff::new(x);

    let y = x.mul(a);


}