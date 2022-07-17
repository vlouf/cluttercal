#include <bom/io/configuration.h>
#include <bom/io/cf.h>
#include <bom/io/nc.h>
#include <bom/io/odim.h>
#include <bom/radar/beam_propagation.h>
#include <bom/array2.h>

#include <getopt.h>
#include <filesystem>
#include <fstream>

using namespace bom;


constexpr auto example_config =
R"(# example layered-flow config

# domain projection
proj4 "+proj=aea +lat_1=-32.2 +lat_2=-35.2 +lon_0=151.209 +lat_0=-33.7008 +a=6378137 +b=6356752.31414 +units=m"

# grid size
size "301 301"
)";


int main(int argc, char* argv[])
{
  try
  {
    // process command line
    while (true)
    {
      int option_index = 0;
      int c = getopt_long(argc, argv, short_options, long_options, &option_index);
      if (c == -1)
        break;
      switch (c)
      {
      case 'h':
        std::cout << usage_string;
        return EXIT_SUCCESS;
      case 'g':
        std::cout << example_config;
        return EXIT_SUCCESS;
      case 't':
        trace::set_min_level(from_string<trace::level>(optarg));
        break;
      case '?':
        std::cerr << try_again;
        return EXIT_FAILURE;
      }
    }

    if (argc - optind != 4)
    {
      std::cerr << "missing required parameter\n" << try_again;
      return EXIT_FAILURE;
    }

    auto config = io::configuration{std::ifstream{argv[optind+0]}};
    if(check_configuration_file(config) != true)
      return EXIT_FAILURE;

    track_layers(
          config
        , argv[optind+1]
        , argv[optind+2]
        , argv[optind+3]
        );
  }
  catch (std::exception& err)
  {
    trace::error("fatal exception: {}", format_exception(err));
    return EXIT_FAILURE;
  }
  catch (...)
  {
    trace::error("fatal exception: (unknown exception)");
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
