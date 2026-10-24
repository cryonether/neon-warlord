//! RigidBody implementation using XPBD

use super::*;

#[derive(Clone)]
pub struct XpbdRigidBody {
    pub p: PositionalQuantities,
    pub r: RotationalQuantities,
}

impl XpbdRigidBody {
    pub fn get_velocity_at(&self, pos: &Vec3) -> Vec3 {
        let p = &self.p;
        let r = &self.r;
        
        if p.inv_mass <= 0.0 {
            return Vec3::zero();
        } 

        p.vel - (pos - p.pos).cross(r.omega)
    }

    pub fn local_to_world(&self, local_pos: &Vec3) -> Vec3 {
        let p = &self.p;
        let r = &self.r;

        r.rot * local_pos + p.pos
    }

    pub fn world_to_local(&self, world_pos: &Vec3) -> Vec3 {
        let p = &self.p;
        let r = &self.r;

        r.inv_rot * (*world_pos - p.pos)
    }
}