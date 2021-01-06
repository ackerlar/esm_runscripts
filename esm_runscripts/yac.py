class yac:
    """

    Generates the configuration file for YAC coupler.
    """

    def __init__(self, full_config, nb_of_couplings=1, coupled_models=["echam", "fesom"], grids=["atmo", "feom"], runtime=1):

        self.name = "yac"

        self.namcouple = ['<?xml version="1.0" encoding="UTF-8"?>']
        self.namcouple += ["<!-- This coupling.xml was automatically generated by the esm-tools (Python) -->"]
        self.namcouple += ["<coupling"]
        self.namcouple += ['\txmlns="http://www.w3schools.com"']
        self.namcouple += ['\txmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"']
        self.namcouple += ['\txsi:schemaLocation="http://www.w3schools.com coupling.xsd">']
        self.namcouple += ['\t<redirect redirect_of_root="' + str(full_config[self.name]['redirect_of_root']).lower() + '" redirect_stdout="' + str(full_config[self.name]['redirect_stdout']).lower() + '"/>']
        self.namcouple += ['\t<components>']
        component_id = 1
        for component in full_config[self.name]["component_names"]:
            self.namcouple += ['\t\t<component id="' + str(component_id) + '">']
            self.namcouple += ['\t\t\t<name>' + str(component) + '</name>']
            self.namcouple += ['\t\t\t<model>' + str(component) + '</model>']
            self.namcouple += ['\t\t\t<simulated>' + str(full_config[coupled_models[component_id - 1]]["type"]) + '</simulated>']
            self.namcouple += ['\t\t\t<transient_grid_refs>']
            for field in range(nb_of_couplings):
                self.namcouple += ['\t\t\t\t<transient_grid_ref id="' + str(field + 1) + '" transient_ref="' + str(field + 1) + '" grid_ref="' + str(component_id) + '" collection_size="1"/>']
            self.namcouple += ['\t\t\t</transient_grid_refs>']
            self.namcouple += ['\t\t</component>']
            component_id += 1
        self.namcouple += ['\t</components>']
        self.namcouple += ['\t<transients>']
        for field in range(nb_of_couplings):
            self.namcouple += ['\t\t<transient id="' + str(field + 1) + '" transient_standard_name="??transient_name' + str(field + 1) + '??"/>']  # Placeholder ??..?? will be replace further down.
        self.namcouple += ['\t</transients>']
        self.namcouple += ['\t<grids>']
        grid_id = 1
        for grid in grids:
            self.namcouple += ['\t\t<grid id="' + str(grid_id) + '" alias_name="' + str(grid) + '" />']
            grid_id += 1
        self.namcouple += ['\t</grids>']
        self.namcouple += ['\t<dates>']
        self.namcouple += ['\t\t\t<start_date>' + str(full_config["general"]['initial_date']) + '</start_date>']
        self.namcouple += ['\t\t\t<end_date>' + str(full_config["general"]['next_date']) + '</end_date>']
        self.namcouple += ['\t\t\t<calendar>' + str(full_config[self.name]['calendar']) + '</calendar>']
        self.namcouple += ['\t</dates>']
        self.namcouple += ['\t<timestep_unit>' + str(full_config[self.name]['coupling_time_step_unit']) + '</timestep_unit>']
        self.namcouple += ["\t<couples>"]
        self.namcouple += ['\t\t<couple>']
        component_id = 1
        for component in coupled_models:
            self.namcouple += ['\t\t\t<component' + str(component_id) + ' component_id="' + str(component_id) + '" />']
            component_id += 1

        self.next_coupling = 1

    def add_coupling(self, field, transient_id, direction, config):
        import sys

        coupling_field = field.replace("<--", "%").replace("--", "&")
        source, rest = coupling_field.split("%")
        source = source.strip()
        interpolation, target = rest.split("&")
        target = target.strip()
        interpolation = interpolation.strip()

        # Replacing placeholder ??..?? with field keyname source
        matching = [s for s in self.namcouple if "??transient_name" + str(transient_id) + "??" in s]
        index = self.namcouple.index(matching[0])
        self.namcouple[index] = self.namcouple[index].replace("??transient_name" + str(transient_id) + "??", source)

        self.namcouple += ['\t\t\t\t<transient_couple transient_id="' + str(transient_id) + '">']
        self.namcouple += ['\t\t\t\t\t<source component_ref="' + str(config["coupling_directions"][direction]["source"]["component_id"]) + '" transient_grid_ref="' + str(transient_id) + '" />']
        self.namcouple += ['\t\t\t\t\t<target transient_grid_ref="' + str(transient_id) + '" />']
        self.namcouple += ['\t\t\t\t\t<timestep>']
        self.namcouple += ['\t\t\t\t\t\t<source>' + str(config["coupling_directions"][direction]["source"]["timestep"]) + '</source>']
        self.namcouple += ['\t\t\t\t\t\t<target>' + str(config["coupling_directions"][direction]["target"]["timestep"]) + '</target>']
        self.namcouple += ['\t\t\t\t\t\t<coupling_period operation="' + str(config["coupling_directions"][direction]["operation"]) + '">' + str(config["coupling_time_step"]) + '</coupling_period>']
        self.namcouple += ['\t\t\t\t\t\t<source_timelag>' + str(config["coupling_directions"][direction]["source"]["timelag"]) + '</source_timelag>']
        self.namcouple += ['\t\t\t\t\t\t<target_timelag>' + str(config["coupling_directions"][direction]["target"]["timelag"]) + '</target_timelag>']
        self.namcouple += ['\t\t\t\t\t</timestep>']
        self.namcouple += ['\t\t\t\t\t<interpolation_requirements use_source_mask="' + str(config["coupling_directions"][direction]["source"]["use_mask"]).lower() + '" use_target_mask="' + str(config["coupling_directions"][direction]["source"]["use_mask"]).lower() + '">']

        line = '<interpolation method="' + str(interpolation) + '" '

        for option in config["interpolation_methods"][interpolation]["options"]:
            value = config["interpolation_methods"][interpolation]["options"][option]
            line += ' ' + str(option) + ' ="' + str(value) + '"'
        line += '/>'
        self.namcouple += ['\t\t\t\t\t\t\t' + line]

        self.namcouple += ['\t\t\t\t\t</interpolation_requirements>']
        self.namcouple += ['\t\t\t\t\t<debug_mode at_source_before_interpolation="' + str(config["coupling_directions"][direction]["source"]["debug_before_interpolation"]).lower() + '" at_source_after_interpolation="' + str(config["coupling_directions"][direction]["source"]["debug_before_interpolation"]).lower() + '" at_target="' + str(config["coupling_directions"][direction]["target"]["debug"]).lower() + '"/>']
        self.namcouple += ['\t\t\t\t\t<enforce_write_restart>' + str(config["coupling_directions"][direction]["write_restart"]).lower() + '</enforce_write_restart>']
        self.namcouple += ['\t\t\t\t\t<enforce_write_weight_file filename="' + str(source) + '_weight">' + str(config["coupling_directions"][direction]["write_weight"]).lower() + '</enforce_write_weight_file>']
        self.namcouple += ['\t\t\t\t</transient_couple>']

    def print_config_files(self):
        for line in self.namcouple:
            print(line)

    def add_output_file(self, lefts, rights, leftmodel, rightmodel, config):
        out_file = []

        coupling = self.next_coupling

        if self.next_coupling < 10:
            this_coupling = "0" + str(coupling)
        else:
            this_coupling = str(coupling)

        for lefty in lefts:
            out_file.append(lefty + "_" + leftmodel + "_" + this_coupling + ".nc")
        for righty in rights:
            out_file.append(righty + "_" + rightmodel + "_" + this_coupling + ".nc")

        self.next_coupling += 1

        if "outdata_files" not in config:
            config["outdata_files"] = {}
        if "outdata_in_work" not in config:
            config["outdata_in_work"] = {}
        if "outdata_sources" not in config:
            config["outdata_sources"] = {}

        for thisfile in out_file:

            config["outdata_files"][thisfile] = thisfile
            config["outdata_in_work"][thisfile] = thisfile
            config["outdata_sources"][thisfile] = thisfile

    def add_restart_files(self, restart_file, fconfig):
        config = fconfig[self.name]
        gconfig = fconfig["general"]
        # enddate = "_" + str(gconfig["end_date"].year) + str(gconfig["end_date"].month) + str(gconfig["end_date"].day)
        # parentdate = "_" + str(config["parent_date"].year) + str(config["parent_date"].month) + str(config["parent_date"].day)
        enddate = "_" + gconfig["end_date"].format(
            form=9, givenph=False, givenpm=False, givenps=False
        )
        parentdate = "_" + config["parent_date"].format(
            form=9, givenph=False, givenpm=False, givenps=False
        )

        if "restart_out_files" not in config:
            config["restart_out_files"] = {}
        if "restart_out_in_work" not in config:
            config["restart_out_in_work"] = {}
        if "restart_out_sources" not in config:
            config["restart_out_sources"] = {}

        if "restart_in_files" not in config:
            config["restart_in_files"] = {}
        if "restart_in_in_work" not in config:
            config["restart_in_in_work"] = {}
        if "restart_in_sources" not in config:
            config["restart_in_sources"] = {}

        config["restart_out_files"][restart_file] = restart_file
        config["restart_out_files"][restart_file + "_recv"] = restart_file + "_recv"

        config["restart_out_in_work"][restart_file] = restart_file  # + enddate
        config["restart_out_in_work"][restart_file + "_recv"] = restart_file + "_recv"  # + enddate

        config["restart_out_sources"][restart_file] = restart_file
        config["restart_out_sources"][restart_file + "_recv"] = restart_file + "_recv"

        config["restart_in_files"][restart_file] = restart_file
        config["restart_in_in_work"][restart_file] = restart_file
        if restart_file not in config["restart_in_sources"]:
            config["restart_in_sources"][restart_file] = restart_file

    def prepare_restarts(self, restart_file, all_fields, model, config):
        enddate = "_" + config["general"]["end_date"].format(
            form=9, givenph=False, givenpm=False, givenps=False
        )
        # enddate = "_" + str(config["general"]["end_date"].year) + str(config["general"]["end_date"].month) + str(config["general"]["end_date"].day)
        import glob
        import os
        import subprocess
        print("Preparing YAC restart files from initial run...")
        exe = config[model]["executable"]
        print(restart_file, all_fields, model, exe)
        cwd = os.getcwd()
        os.chdir(config["general"]["thisrun_work_dir"])
        filelist = ""
        for field in all_fields:
            print(field + "-" + model)
            thesefiles = glob.glob(field + "_" + exe + "_*.nc")
            print(thesefiles)
            for thisfile in thesefiles:
                print("cdo showtime " + thisfile + " 2>/dev/null | wc -w")
                lasttimestep = subprocess.check_output("cdo showtime " + thisfile + " 2>/dev/null | wc -w", shell=True).decode("utf-8").rstrip()
                # print (lasttimestep)

                print("cdo -O seltimestep," + str(lasttimestep) + " " + thisfile + " onlyonetimestep.nc")
                os.system("cdo -O seltimestep," + str(lasttimestep) + " " + thisfile + " onlyonetimestep.nc")
                print("ncwa -O -a time onlyonetimestep.nc notimestep_" + field + ".nc")
                os.system("ncwa -O -a time onlyonetimestep.nc notimestep_" + field + ".nc")
                filelist += "notimestep_" + field + ".nc "
                print(filelist)
        print("cdo merge " + filelist + " " + restart_file)  # + enddate)
        os.system("cdo merge " + filelist + " " + restart_file)  # + enddate)
        rmlist = glob.glob("notimestep*")
        rmlist.append("onlyonetimestep.nc")
        for rmfile in rmlist:
            print("rm " + rmfile)
            os.system("rm " + rmfile)
        os.chdir(cwd)

    def finalize(self, destination_dir):
        self.namcouple += ["\t\t</couple>"]
        self.namcouple += ["\t</couples>"]
        self.namcouple += ["</coupling>"]
        endline = ""
        with open(destination_dir + "/coupling.xml", "w") as namcouple:
            for line in self.namcouple:
                namcouple.write(endline)
                namcouple.write(line)
                endline = "\n"
