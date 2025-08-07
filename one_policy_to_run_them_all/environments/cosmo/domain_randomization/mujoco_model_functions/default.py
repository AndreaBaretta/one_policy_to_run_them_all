import mujoco

class DefaultDomainMuJoCoModel:
    def __init__(self, env,
                 friction_tangential_min=0.8, friction_tangential_max=1.2,
                 friction_torsional_ground_min=0.003, friction_torsional_ground_max=0.007,
                 friction_torsional_feet_min=0.003, friction_torsional_feet_max=0.007,
                 friction_rolling_ground_min=0.00008, friction_rolling_ground_max=0.00012,
                 friction_rolling_feet_min=0.00008, friction_rolling_feet_max=0.00012,
                 damping_min=72, damping_max=88,
                 stiffness_min=900, stiffness_max=1100,
                 gravity_min=9.51, gravity_max=10.11,
        ):
        self.env = env
        self.friction_tangential_min = friction_tangential_min
        self.friction_tangential_max = friction_tangential_max
        self.friction_torsional_ground_min = friction_torsional_ground_min
        self.friction_torsional_ground_max = friction_torsional_ground_max
        self.friction_torsional_feet_min = friction_torsional_feet_min
        self.friction_torsional_feet_max = friction_torsional_feet_max
        self.friction_rolling_ground_min = friction_rolling_ground_min
        self.friction_rolling_ground_max = friction_rolling_ground_max
        self.friction_rolling_feet_min = friction_rolling_feet_min
        self.friction_rolling_feet_max = friction_rolling_feet_max
        self.damping_min = damping_min
        self.damping_max = damping_max
        self.stiffness_min = stiffness_min
        self.stiffness_max = stiffness_max
        self.gravity_min = gravity_min
        self.gravity_max = gravity_max
    
    def init(self):
        self.sampled_friction_tangential = self.env.model.geom_friction[0, 0]
        self.sampled_friction_torsional_ground = self.env.model.geom_friction[0, 1]
        # In Cassie, 44 is right_foot, 24 is left_foot
        # print([mujoco.mj_id2name(self.env.model, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(self.env.model.ngeom)])
        # import pdb; pdb.set_trace()
        # In Cosmo: geom_names - ['body_collision', 'body_visual', 'hip_l_visual', 'adductor_l_collision', 'adductor_l_visual', 'femur_l_collision', 'femur_l_visual', 'tibia_l_collision', 'tibia_l_visual', 'foot_l_collision', 'foot_l_visual', 'hip_r_visual', 'adductor_r_collision', 'adductor_r_visual', 'femur_r_collision', 'femur_r_visual', 'tibia_r_collision', 'tibia_r_visual', 'foot_r_collision', 'foot_r_visual', 'clavicle_l_visual', 'upperarm_l_visual', 'forearm_l_collision', 'forearm_l_visual', 'hand_l_collision', 'hand_l_visual', 'clavicle_r_visual', 'upperarm_r_visual', 'forearm_r_collision', 'forearm_r_visual', 'hand_r_collision', 'hand_r_visual', 'neck_bottom_visual', 'neck_top_collision']
        # 9 is foot_l_collision
        # 18 is foot_r_collision
        self.sampled_friction_torsional_feet = self.env.model.geom_friction[18, 1]
        self.sampled_friction_rolling_ground = self.env.model.geom_friction[0, 2]
        self.sampled_friction_rolling_feet = self.env.model.geom_friction[18, 2]
        self.sampled_damping = -self.env.model.geom_solref[0, 1]
        self.sampled_stiffness = -self.env.model.geom_solref[0, 0]
        self.sampled_gravity = -self.env.model.opt.gravity[2]

    def sample(self):
        interpolation = self.env.np_rng.uniform(0, 1)
        self.sampled_friction_tangential = self.friction_tangential_min + (self.friction_tangential_max - self.friction_tangential_min) * interpolation
        self.sampled_friction_torsional_ground = self.friction_torsional_ground_min + (self.friction_torsional_ground_max - self.friction_torsional_ground_min) * interpolation
        self.sampled_friction_torsional_feet = self.friction_torsional_feet_min + (self.friction_torsional_feet_max - self.friction_torsional_feet_min) * interpolation
        self.sampled_friction_rolling_ground = self.friction_rolling_ground_min + (self.friction_rolling_ground_max - self.friction_rolling_ground_min) * interpolation
        self.sampled_friction_rolling_feet = self.friction_rolling_feet_min + (self.friction_rolling_feet_max - self.friction_rolling_feet_min) * interpolation
        self.env.model.geom_friction[0] = [self.sampled_friction_tangential, self.sampled_friction_torsional_ground, self.sampled_friction_rolling_ground]
        self.env.model.geom_friction[[9, 18]] = [self.sampled_friction_tangential, self.sampled_friction_torsional_feet, self.sampled_friction_rolling_feet]

        interpolation = self.env.np_rng.uniform(0, 1)
        self.sampled_damping = self.damping_min + (self.damping_max - self.damping_min) * interpolation
        self.sampled_stiffness = self.stiffness_min + (self.stiffness_max - self.stiffness_min) * interpolation
        self.env.model.geom_solref[:, 0] = -self.sampled_stiffness
        self.env.model.geom_solref[:, 1] = -self.sampled_damping

        self.sampled_gravity = self.env.np_rng.uniform(self.gravity_min, self.gravity_max)
        self.env.model.opt.gravity[2] = -self.sampled_gravity
