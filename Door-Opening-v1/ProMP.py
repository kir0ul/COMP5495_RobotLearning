import json
import os
import shutil
import time

import h5py
import numpy as np
import matplotlib.pyplot as plt
from movement_primitives.promp import ProMP
from scipy import interpolate
from scipy.interpolate import interp1d

extension = '.h5'
currentDemo = -1
dof = 6

for filename in os.listdir():
    if filename.endswith(extension) and filename != 'traj.h5':
        print(filename)
        with h5py.File(filename, "r") as f:
            print("Reading " + filename)

            num_demos = list(f.keys()).__len__()
            timesteps = f['demo1']['timestamps'].shape[0]
            xmlFile = f['demo1'].attrs['xmlmodel']

            stateSize = f['demo1']['states'].shape[1]
            actionSize = f['demo1']['actions'].shape[1]
            T = np.zeros([num_demos, timesteps])
            Y = np.zeros([num_demos, timesteps, stateSize])
            #States = np.zeros([num_demos, timesteps, stateSize])#fileStates.shape])
            Actions = np.zeros([num_demos, timesteps, actionSize])

            for i in range(num_demos):
                currentDemo = currentDemo + 1
                fileTimestamps = f['demo' + str(currentDemo + 1)]['timestamps']
                T[currentDemo] = fileTimestamps

                fileStates = f['demo' + str(currentDemo + 1)]['states']
                fileActions = f['demo' + str(currentDemo + 1)]['actions']
                #States = fileStates[:][:]
            #print(fileStates[0])
            #print(fileStates[0][1:4])
            #print(fileStates[0][1:4].shape)
                for j in range(fileStates.shape[0]):
                    Y[currentDemo][j] = fileStates[j]
                    Actions[currentDemo][j] = fileActions[j]

                #Actions = fileActions[:][:]

                fileTime = fileStates[0]
                fileQPOS = fileStates[:,1:27]
                fileQVEL = fileStates[27:]

            #print(fileStates[:,1:27].shape)
            #print(fileActions)

#States = np.zeros([timesteps, stateSize])
#Actions = np.zeros([timesteps, actionSize])
genStates = np.zeros([timesteps, stateSize])
genActions = np.zeros([timesteps, actionSize])
genStates[:,:] = Y[0,:,:]
genActions = Actions

y_conditional_cov = np.array([0.025])
promp = ProMP(n_dims=3, n_weights_per_dim=10)
promp.imitate(T, Y[:,:,1:4])

#weights = promp.weights(T[0], Y[0,:,0:3])
weights = promp.weights(T[0], Y[0,:,1:4])

promp.trajectory_from_weights(T[0], weights)
Y_mean = promp.mean_trajectory(T[0])
Y_conf = 1.96 * np.sqrt(promp.var_trajectory(T[0]))
#States[:,0] = T[0]
#States[:,1:4] = Y_mean
genStates[:,0] = T[0]
genStates[:,1:4] = Y_mean

fig = plt.figure(figsize=(10, 5))

ax1 = fig.add_subplot(111, projection='3d')
for x in range(num_demos):
    ax1.plot(T[x], Y[x, :, 1], Y[x, :, 2], Y[x, :, 3], label="Demo %g" %(x+1))

ax1.plot(T[0], Y_mean[:, 0], Y_mean[:, 1], Y_mean[:, 2], label="ProMP Mean Trajectory")
ax1.legend(loc="best")
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')
ax1.set_title('Demo Trajectories Compared to ProMP Mean Trajectory')
plt.show()

fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(10, 7.5),layout="constrained")
for x in range(num_demos):
    axs[0][0].plot(T[x], Y[x, :, 1], label="Demo %g" %(x+1))
    axs[0][0].set_title('x')
    axs[0][1].plot(T[x], Y[x, :, 2], label="Demo %g" % (x + 1))
    axs[0][1].set_title('y')
    axs[1][0].plot(T[x], Y[x, :, 3], label="Demo %g" % (x + 1))
    axs[1][0].set_title('z')

axs[0][0].plot(T[0], Y_mean[:, 0], label="ProMP Mean Trajectory")
axs[0][1].plot(T[0], Y_mean[:, 1], label="ProMP Mean Trajectory")
axs[1][0].plot(T[0], Y_mean[:, 2], label="ProMP Mean Trajectory")
axs[0][0].legend(loc="best")
axs[0][1].legend(loc="best")
axs[1][0].legend(loc="best")


for i in range(1, dof):
    y_conditional_cov = np.array([0.025])
    extrapromp = ProMP(n_dims=3, n_weights_per_dim=10)
    extrapromp.imitate(T, Y[:,:,3*i+1:3*i+4])

    weights = extrapromp.weights(T[0], Y[0,:,3*i+1:3*i+4])

    extrapromp.trajectory_from_weights(T[0], weights)
    Y_mean = extrapromp.mean_trajectory(T[0])
    Y_conf = 1.96 * np.sqrt(extrapromp.var_trajectory(T[0]))
    #States[:,3*i+1:3*i+4] = Y_mean
    genStates[:, 3 * i + 1:3 * i + 4] = Y_mean

extension = 'traj.h5'
for filename in os.listdir():
    if filename.endswith(extension):
        os.remove(filename)

if(os.path.isdir('ep_traj')):
    shutil.rmtree('ep_traj')

#t1, t2 = str(time.time()).split(".")
#self.ep_directory = os.path.join(self.directory, "ep_{}_{}".format(t1, t2))
#assert not os.path.exists(self.ep_directory)
#os.makedirs(self.ep_directory)
os.makedirs("ep_traj")

xml_path = os.path.join("ep_traj", "model.xml")
with open(xml_path, "w") as f:
    f.write(xmlFile)

        # save the episode info to json file
#ep_meta_path = os.path.join(self.ep_directory, "ep_meta.json")
#with open(ep_meta_path, "w") as f:
    #json.dump(self.env.get_ep_meta(), f)

# save initial state and action
#assert len(self.states) == 0
#env.sim.model.get_xml()
#self.states.append(self._current_task_instance_state)

np.savez(
            'ep_traj/ep_traj',
            states=np.array(genStates),
            action_infos=Actions,
            #successful=self.successful,
            env='Door',
        )

returnF = h5py.File("traj.h5", "w")
grp = returnF.create_group("genTraj")
grp.create_dataset("timestamps", data=T[0])
grp.create_dataset("states", data=genStates)
grp.create_dataset("actions", data=Actions)
grp.attrs['episode'] = "ep_traj"
grp.attrs['max_fr'] = 20
plt.show()
