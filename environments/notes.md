Good question to continually ask.

Can the state describe the information needed for a good decision?
Does the reward encourage the behavior we actually want?
Has the agent had enough relevant experience to learn it?

Sensor to consider for observation space. 

RGB cameras, Radar, Infared, LiDar, Event-Camera 

Estimate are at 5-10 watts per camera + hvac considerations 

Potential weather considerations that effect sensor observation performance 

Rain, Fog, Bright Sun, Snow

Weather effects on sensors

Rain = reduced visibility, signal scattering, radar inference, weak reflection 
Fog = blurred image, limited range, attenuated signals, poor detection
snow = lens obstruction, snow reflection, signal attenuation, icy sensors, lidar sensors
bright sun = glare and shadow, heat distribution, false returns, overheating 
night = low visibility for passive sensors

Configuration of most SAE AV level 5 is estimated at 35+ sensors. For this experiment we will reduce that. Some additional consideration are sensor configuration, this will be left out intitially to elimate too many moving targets. 

Can an agent learn when extra sensor information is worth its energy cost, especially as weather changes?

True traffic + weather + sensor configuration
                      ↓
         Simulated sensor measurements
                      ↓
          Agent chooses driving and sensing actions

             N: forward
        NW               NE

 W: left       CAR       E: right

        SW               SE
              S: rear

I want to try to model these sensor close to reality although we must consider this is somewhat a blind experiment as we will be making assumption about what sensor can and can't do. I do want to base this on know literature and logic though. For example, the earlier paper established radar is much better with a RGB camera. So we must insert some level of uncertainty penality into the environment. For example, the use of radar and rgb camera should give a more complete picture. For the sensor view I want to place event-camera and rgb in 8 directional areas going 360 degrees binned by cardial direction like N which will always be the vehicle forward view and S will always be the rear view. I want a grid to represent distance up to 150m with each cell representing 2 meter intervals. LIDAR can have 360 degree access. I also want weather to effect sensor differently. For example I believe radar is more accurate in rain, event-camera are known to be better at capturing object in blinding light and in lower-visibility. So we need to make some assumption about these camera based on known literature.

| Condition | RGB | Event camera | Radar | LiDAR |
|---|---|---|---|---|
| Strong illumination contrast | Increased missed detections in affected areas | Smaller degradation than RGB under suitable conditions | Little direct illumination effect | Device-dependent sunlight interference |
| Very low light | Reduced useful detections | Still limited by available light and contrast | Little direct illumination effect | Active ranging remains available |
| Fog | Contrast and useful range decrease | Model optical attenuation; no assumed fog advantage | Generally more robust than optical sensing | Reduced returns and backscatter |
| Rain | Visibility loss and lens obstruction | Optical degradation; possible distracting events | Possible clutter and attenuation | Lost returns and spurious returns |


| Module | Responsibility | Example changes |
|---|---|---|
| Simulator adapter | Advances traffic, applies driving commands, provides internal ground truth | HighwayEnv now; CARLA later |
| Sensor models | Converts ground truth into imperfect, delayed measurements | Range, field of view, noise, power, mounting direction |
| Weather model | Defines conditions and their effects on sensors | Fog visibility, rain intensity, directional glare |
| Perception and fusion | Combines available measurements and tracks uncertainty | Radar–RGB fusion, tracking, measurement aging |
| Observation builder | Produces the agent’s grid and sensor-status information | Grid extent, cell size, observation layers |
| Reward model | Scores driving, energy consumption, and relevant uncertainty | Change weights or disable uncertainty penalties |
| Gymnasium environment | Connects these modules through `reset()` and `step()` | Supports training algorithms without exposing simulator details |