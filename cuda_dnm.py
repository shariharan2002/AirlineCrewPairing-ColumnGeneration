
import cupy as cp
import numpy as np
import pickle
def write_pickle_cuda(data,filename):
    #print("Writing --> "+str(filename))
    file_path=r"D:\dnm_files/"+str(filename)
    file_path=file_path
    with open(file_path, 'wb') as file:
        # Serialize and write the variable to the file
        pickle.dump(data, file)
    #print(str(filename)+"--> Saved !!!")

class DNM_creator_via_cudacupy:
    def __init__(self,CUDA_container,version,cuda_increase_param=500):
        self.kernel_code=self.read_kernel_code()
        self.crew_bases=CUDA_container.crew_bases
        self.legs=CUDA_container.legs
        self.all_airports=CUDA_container.all_airports
        self.airport_arr_flights=CUDA_container.airport_arr_flights
        self.airport_dep_flights=CUDA_container.airport_dep_flights
        self.adjacency_matrix_of_connections=CUDA_container.adjacency_matrix_of_connections
        self.leg_to_index_mapper=CUDA_container.leg_to_index_mapper
        self.index_to_leg_mapper=CUDA_container.index_to_leg_mapper
        self.dg=CUDA_container.dg
        self.crew_bases_indices=CUDA_container.crew_bases_indices
        if version==1:
            self.duty_network_matrix=self.calculate_duty_network_matrix(self.kernel_code,self.crew_bases,self.legs,self.all_airports,self.airport_arr_flights,self.airport_dep_flights,self.adjacency_matrix_of_connections,self.leg_to_index_mapper,self.index_to_leg_mapper,self.dg,self.crew_bases_indices,cuda_increase_param)
        elif version==2:
            self.filename_list=self.calculate_duty_network_matrix_v2(self.kernel_code,self.crew_bases,self.legs,self.all_airports,self.airport_arr_flights,self.airport_dep_flights,self.adjacency_matrix_of_connections,self.leg_to_index_mapper,self.index_to_leg_mapper,self.dg,self.crew_bases_indices,cuda_increase_param)
    
    def read_kernel_code(self):
        content=""
        with open("kernel_code.txt",'r') as file:
            content=file.read()
        return content
    
    def calculate_duty_network_matrix(self,kernel_code,crew_bases,legs,all_airports,airport_arr_flights,airport_dep_flights,adjacency_matrix_of_connections,leg_to_index_mapper,index_to_leg_mapper,dg,crew_bases_indices,cuda_increase_param=500):

        airport_to_index_mapper_cuda={}
        index_to_airport_mapper_cuda={}
        for i in range(len(all_airports)):
            airport_to_index_mapper_cuda[all_airports[i]]=i
            index_to_airport_mapper_cuda[i]=all_airports[i]

        duty_origins_of_first_flights_cuda=[]
        duty_destinations_of_last_flights_cuda=[]
        duty_dep_times_of_first_flights_cuda=[]
        duty_arr_times_of_last_flights_cuda=[]

        crew_bases_indices_cuda_mapper={}
        for k,v in crew_bases_indices.items():
            try:
                crew_bases_indices_cuda_mapper[airport_to_index_mapper_cuda[k]]=v
            except:
                pass

        crew_bases_indices_involved_cuda=[]
        for k,v in crew_bases_indices_cuda_mapper.items():
            crew_bases_indices_involved_cuda.append(k)
        crew_bases_indices_involved_cuda=sorted(crew_bases_indices_involved_cuda)

        crew_bases_indices_mapper_cuda_list=[]
        for k,v in airport_to_index_mapper_cuda.items():
            crew_bases_indices_mapper_cuda_list.append([-1,-1])
        for k in crew_bases_indices_involved_cuda:
            crew_bases_indices_mapper_cuda_list[k]=crew_bases_indices_cuda_mapper[k]
        temp_cuda=[]
        for item in crew_bases_indices_mapper_cuda_list:
            temp_cuda.append(item[0])
            temp_cuda.append(item[1])

        crew_bases_indices_mapper_cuda_list=temp_cuda.copy()

        for i in range(len(dg.duty_list)):
            first_flight=dg.duty_list[i].duty[0]
            last_flight=dg.duty_list[i].duty[-1]
            duty_origins_of_first_flights_cuda.append(airport_to_index_mapper_cuda[first_flight.origin])
            duty_destinations_of_last_flights_cuda.append(airport_to_index_mapper_cuda[last_flight.destination])
            duty_dep_times_of_first_flights_cuda.append(first_flight.dep_time)
            duty_arr_times_of_last_flights_cuda.append(last_flight.arr_time)

        crew_bases_indices_in_order=[]
        for k,v in crew_bases_indices.items():
            crew_bases_indices_in_order.append(airport_to_index_mapper_cuda[k])

        for i in range(len(crew_bases_indices_in_order)):
            duty_origins_of_first_flights_cuda.append(crew_bases_indices_in_order[i])
            duty_destinations_of_last_flights_cuda.append(crew_bases_indices_in_order[i])

        for i in range(len(crew_bases_indices_in_order)):
            duty_origins_of_first_flights_cuda.append(crew_bases_indices_in_order[i])
            duty_destinations_of_last_flights_cuda.append(crew_bases_indices_in_order[i])

        origins = cp.array(np.array(duty_origins_of_first_flights_cuda),dtype=cp.int32)
        destinations= cp.array(np.array(duty_destinations_of_last_flights_cuda),dtype=cp.int32)
        dep_times=cp.array(np.array(duty_dep_times_of_first_flights_cuda),dtype=cp.int32)
        arr_times=cp.array(np.array(duty_arr_times_of_last_flights_cuda),dtype=cp.int32)
        crew_list=cp.array(np.array(crew_bases_indices_involved_cuda),dtype=cp.int32)
        crew_indices=cp.array(np.array(crew_bases_indices_mapper_cuda_list),dtype=cp.int32)
        threshold_size=len(dg.duty_list)
        crew_list_size=len(crew_bases)



        duty_network_size=len(dg.duty_list)+(2*len(crew_bases))
        cuda_increase_param=500

        duty_network_matrix=[]

        l_ptr=0
        e_ptr=min(cuda_increase_param,duty_network_size)

        dn_kernel = cp.RawKernel(kernel_code, 'form_duty_network_kernel')
        list_of_output_arrays_cpu=[]



        while l_ptr<duty_network_size:
            cuda_start_range=l_ptr
            cuda_end_range=e_ptr
            array_size=cuda_end_range-cuda_start_range
            output_array = cp.empty(duty_network_size*array_size, dtype=cp.bool_)
            array_start_size=cuda_start_range
            array_end_size=cuda_end_range
            
            partition_size=duty_network_size

            block_size = (1,)
            grid_size = (duty_network_size,array_size,)
            
            dn_kernel(grid_size, block_size, (origins, destinations, dep_times, arr_times, crew_list, crew_indices, output_array,crew_list_size, threshold_size, array_start_size,array_end_size,partition_size, 8*60))

            output_array_cpu = output_array.get()
            output_array=None
            list_of_output_arrays_cpu.append((output_array_cpu,cuda_start_range,cuda_end_range))

            for k in range(cuda_increase_param):
                duty_network_matrix.append(output_array_cpu[k*partition_size: (k+1)*partition_size].copy())

            l_ptr=e_ptr
            if duty_network_size-e_ptr>cuda_increase_param:
                e_ptr+=cuda_increase_param
            else:
                e_ptr+=duty_network_size-e_ptr



        for k in range(len(dg.duty_list)):
            destination=dg.duty_list[k].duty[-1].destination
            if destination in crew_bases_indices.keys():
                corres_crew_base_dest_index=crew_bases_indices[destination][1]
                duty_network_matrix[k][corres_crew_base_dest_index]=True
            origin=dg.duty_list[k].duty[0].origin
            if origin in crew_bases_indices.keys():
                corres_crew_base_origin_index=crew_bases_indices[origin][0]
                duty_network_matrix[corres_crew_base_origin_index][k]=True
        print("Duty Network matrix computation via Cuda completed")
        return duty_network_matrix

    def calculate_duty_network_matrix_v2(self,kernel_code,crew_bases,legs,all_airports,airport_arr_flights,airport_dep_flights,adjacency_matrix_of_connections,leg_to_index_mapper,index_to_leg_mapper,dg,crew_bases_indices,cuda_increase_param=500):
        '''
        Here, even the true/false matrix cannot be stored easily in memory. Hence, once we complete each row, we shall store it 
        in secondary storage as pickle file. 
        We will then later on retrieve it and form adjacency lists.
        '''
        airport_to_index_mapper_cuda={}
        index_to_airport_mapper_cuda={}
        for i in range(len(all_airports)):
            airport_to_index_mapper_cuda[all_airports[i]]=i
            index_to_airport_mapper_cuda[i]=all_airports[i]

        duty_origins_of_first_flights_cuda=[]
        duty_destinations_of_last_flights_cuda=[]
        duty_dep_times_of_first_flights_cuda=[]
        duty_arr_times_of_last_flights_cuda=[]

        crew_bases_indices_cuda_mapper={}
        for k,v in crew_bases_indices.items():
            try:
                crew_bases_indices_cuda_mapper[airport_to_index_mapper_cuda[k]]=v
            except:
                pass

        crew_bases_indices_involved_cuda=[]
        for k,v in crew_bases_indices_cuda_mapper.items():
            crew_bases_indices_involved_cuda.append(k)
        crew_bases_indices_involved_cuda=sorted(crew_bases_indices_involved_cuda)

        crew_bases_indices_mapper_cuda_list=[]
        for k,v in airport_to_index_mapper_cuda.items():
            crew_bases_indices_mapper_cuda_list.append([-1,-1])
        for k in crew_bases_indices_involved_cuda:
            crew_bases_indices_mapper_cuda_list[k]=crew_bases_indices_cuda_mapper[k]
        temp_cuda=[]
        for item in crew_bases_indices_mapper_cuda_list:
            temp_cuda.append(item[0])
            temp_cuda.append(item[1])

        crew_bases_indices_mapper_cuda_list=temp_cuda.copy()

        for i in range(len(dg.duty_list)):
            first_flight=dg.duty_list[i].duty[0]
            last_flight=dg.duty_list[i].duty[-1]
            duty_origins_of_first_flights_cuda.append(airport_to_index_mapper_cuda[first_flight.origin])
            duty_destinations_of_last_flights_cuda.append(airport_to_index_mapper_cuda[last_flight.destination])
            duty_dep_times_of_first_flights_cuda.append(first_flight.dep_time)
            duty_arr_times_of_last_flights_cuda.append(last_flight.arr_time)

        crew_bases_indices_in_order=[]
        for k,v in crew_bases_indices.items():
            crew_bases_indices_in_order.append(airport_to_index_mapper_cuda[k])

        for i in range(len(crew_bases_indices_in_order)):
            duty_origins_of_first_flights_cuda.append(crew_bases_indices_in_order[i])
            duty_destinations_of_last_flights_cuda.append(crew_bases_indices_in_order[i])

        for i in range(len(crew_bases_indices_in_order)):
            duty_origins_of_first_flights_cuda.append(crew_bases_indices_in_order[i])
            duty_destinations_of_last_flights_cuda.append(crew_bases_indices_in_order[i])

        origins = cp.array(np.array(duty_origins_of_first_flights_cuda),dtype=cp.int32)
        destinations= cp.array(np.array(duty_destinations_of_last_flights_cuda),dtype=cp.int32)
        dep_times=cp.array(np.array(duty_dep_times_of_first_flights_cuda),dtype=cp.int32)
        arr_times=cp.array(np.array(duty_arr_times_of_last_flights_cuda),dtype=cp.int32)
        crew_list=cp.array(np.array(crew_bases_indices_involved_cuda),dtype=cp.int32)
        crew_indices=cp.array(np.array(crew_bases_indices_mapper_cuda_list),dtype=cp.int32)
        threshold_size=len(dg.duty_list)
        crew_list_size=len(crew_bases)



        duty_network_size=len(dg.duty_list)+(2*len(crew_bases))
        cuda_increase_param=500

        duty_network_matrix=[]

        l_ptr=0
        e_ptr=min(cuda_increase_param,duty_network_size)

        dn_kernel = cp.RawKernel(kernel_code, 'form_duty_network_kernel')
        list_of_output_arrays_cpu=[]


        filename_list=[]
        while l_ptr<duty_network_size:
            cuda_start_range=l_ptr
            cuda_end_range=e_ptr
            array_size=cuda_end_range-cuda_start_range
            output_array = cp.empty(duty_network_size*array_size, dtype=cp.bool_)
            array_start_size=cuda_start_range
            array_end_size=cuda_end_range
            
            partition_size=duty_network_size

            block_size = (1,)
            grid_size = (duty_network_size,array_size,)
            
            dn_kernel(grid_size, block_size, (origins, destinations, dep_times, arr_times, crew_list, crew_indices, output_array,crew_list_size, threshold_size, array_start_size,array_end_size,partition_size, 8*60))

            output_array_cpu = output_array.get()
            output_array=None
            list_of_output_arrays_cpu.append((output_array_cpu,cuda_start_range,cuda_end_range))

            # print(cuda_increase_param,e_ptr-l_ptr)
            # print("Revised Cuda Increase param=",cuda_increase_param)
            for k in range(cuda_increase_param):
                #duty_network_matrix.append(output_array_cpu[k*partition_size: (k+1)*partition_size].copy())
                if k+l_ptr<len(dg.duty_list):
                    destination=dg.duty_list[k+l_ptr].duty[-1].destination
                    if destination in crew_bases_indices.keys():
                        corres_crew_base_dest_index=crew_bases_indices[destination][1]
                        #print((k*partition_size)+corres_crew_base_dest_index)
                        output_array_cpu[(k*partition_size)+corres_crew_base_dest_index]=True
                
                if k+l_ptr>=len(dg.duty_list) and k+l_ptr<len(dg.duty_list)+len(crew_bases):
                    #this means we are dealing with an origin dp....
                    for z in range(len(dg.duty_list)):
                        origin=dg.duty_list[z].duty[0].origin
                        if origin in crew_bases_indices.keys():
                            corres_crew_base_origin_index=crew_bases_indices[origin][0]
                            if k+l_ptr==corres_crew_base_origin_index:
                                output_array_cpu[(k*partition_size)+z]=True
                
                write_pickle_cuda(output_array_cpu[k*partition_size: (k+1)*partition_size].copy(),str(k+l_ptr)+'_DNM.pickle')
                filename_list.append(r"D:\dnm_files/"+ str(k+l_ptr)+'_DNM.pickle')
    
            l_ptr=e_ptr
            # print("Revised l_ptr = ",l_ptr)
            if duty_network_size-e_ptr>cuda_increase_param:
                e_ptr+=cuda_increase_param
            else:
                cuda_increase_param=duty_network_size-e_ptr
                e_ptr+=duty_network_size-e_ptr
        

        print("Duty Network matrix computation via Pickle completed. You shall have received the filenames")
        print("Use the filenames to retrieve the pickle files.")
        return filename_list





    
    

