/**
 ********************************************************************************
 * @file    main.cpp
 * @author  [absadiki](https://github.com/absadiki)
 * @date    2023
 * @brief   Python bindings for [whisper.cpp](https://github.com/ggerganov/whisper.cpp) using Pybind11
 *
 * @par
 * COPYRIGHT NOTICE: (c) 2023.  All rights reserved.
 ********************************************************************************
 */

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <cstring>
#include <sstream>
#include "whisper.h"

namespace nb = nanobind;
using namespace nb::literals;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

nb::object py_new_segment_callback;
nb::object py_encoder_begin_callback;
nb::object py_logits_filter_callback;
nb::object py_whisper_log_callback;

// whisper context wrapper, to solve the incomplete type issue
// Thanks to https://github.com/pybind/pybind11/issues/2770
struct whisper_context_wrapper {
    whisper_context* ptr;
};

// struct inside params
struct greedy{
    int best_of;
};

struct beam_search{
    int beam_size;
    float patience;
};

struct whisper_model_loader_wrapper {
    whisper_model_loader* ptr;

};

struct whisper_context_wrapper whisper_init_from_file_wrapper(const char * path_model){
    struct whisper_context * ctx = whisper_init_from_file_with_params(path_model, whisper_context_default_params());
    struct whisper_context_wrapper ctw_w;
    ctw_w.ptr = ctx;
    return ctw_w;
}

struct whisper_context_wrapper whisper_init_from_buffer_wrapper(void * buffer, size_t buffer_size){
    struct whisper_context * ctx = whisper_init_from_buffer_with_params(buffer, buffer_size, whisper_context_default_params());
    struct whisper_context_wrapper ctw_w;
    ctw_w.ptr = ctx;
    return ctw_w;
}

struct whisper_context_wrapper whisper_init_wrapper(struct whisper_model_loader_wrapper * loader){
    struct whisper_context * ctx = whisper_init_with_params(loader->ptr, whisper_context_default_params());
    struct whisper_context_wrapper ctw_w;
    ctw_w.ptr = ctx;
    return ctw_w;
};

void whisper_free_wrapper(struct whisper_context_wrapper * ctx_w){
    whisper_free(ctx_w->ptr);
};

int whisper_pcm_to_mel_wrapper(
    whisper_context_wrapper * ctx,
    nb::ndarray<nb::numpy, float, nb::shape<-1>, nb::c_contig> samples,
    int n_samples,
    int n_threads) {

    return whisper_pcm_to_mel(ctx->ptr, samples.data(), n_samples, n_threads);
}

int whisper_set_mel_wrapper(
    whisper_context_wrapper * ctx,
    nb::ndarray<nb::numpy, float, nb::shape<-1>, nb::c_contig> data,
    int n_len,
    int n_mel) {

    return whisper_set_mel(ctx->ptr, data.data(), n_len, n_mel);
}

int whisper_n_len_wrapper(struct whisper_context_wrapper * ctx_w){
    return whisper_n_len(ctx_w->ptr);
};

int whisper_n_vocab_wrapper(struct whisper_context_wrapper * ctx_w){
    return whisper_n_vocab(ctx_w->ptr);
};

int whisper_n_text_ctx_wrapper(struct whisper_context_wrapper * ctx_w){
    return whisper_n_text_ctx(ctx_w->ptr);
};

int whisper_n_audio_ctx_wrapper(struct whisper_context_wrapper * ctx_w){
    return whisper_n_audio_ctx(ctx_w->ptr);
}

int whisper_is_multilingual_wrapper(struct whisper_context_wrapper * ctx_w){
    return whisper_is_multilingual(ctx_w->ptr);
}


nb::ndarray<nb::numpy, float> whisper_get_logits_wrapper(
    whisper_context_wrapper * ctx_w,
    size_t n) {

    float *ptr = whisper_get_logits(ctx_w->ptr);
    return nb::ndarray<nb::numpy, float>(
        ptr,
        { n },
        nb::handle()   // no owner; whisper owns memory
    );
}

nb::bytes whisper_token_to_str_wrapper(
    whisper_context_wrapper * ctx_w,
    whisper_token token) {

    const char *str = whisper_token_to_str(ctx_w->ptr, token);
    return str ? nb::bytes(str, strlen(str)) : nb::bytes();
}

nb::bytes whisper_token_to_bytes_wrapper(struct whisper_context_wrapper * ctx_w, whisper_token token){
    const char* str = whisper_token_to_str(ctx_w->ptr, token);
    size_t l = strlen(str);
    return nb::bytes(str, l);
}

whisper_token whisper_token_eot_wrapper(struct whisper_context_wrapper * ctx_w){
    return whisper_token_eot(ctx_w->ptr);
}

whisper_token whisper_token_sot_wrapper(struct whisper_context_wrapper * ctx_w){
    return whisper_token_sot(ctx_w->ptr);
}

whisper_token whisper_token_prev_wrapper(struct whisper_context_wrapper * ctx_w){
    return whisper_token_prev(ctx_w->ptr);
}

whisper_token whisper_token_solm_wrapper(struct whisper_context_wrapper * ctx_w){
    return whisper_token_solm(ctx_w->ptr);
}

whisper_token whisper_token_not_wrapper(struct whisper_context_wrapper * ctx_w){
    return whisper_token_not(ctx_w->ptr);
}

whisper_token whisper_token_beg_wrapper(struct whisper_context_wrapper * ctx_w){
    return whisper_token_beg(ctx_w->ptr);
}

whisper_token whisper_token_lang_wrapper(struct whisper_context_wrapper * ctx_w, int lang_id){
    return whisper_token_lang(ctx_w->ptr, lang_id);
}

whisper_token whisper_token_translate_wrapper(struct whisper_context_wrapper * ctx_w){
    return whisper_token_translate(ctx_w->ptr);
}

whisper_token whisper_token_transcribe_wrapper(struct whisper_context_wrapper * ctx_w){
    return whisper_token_transcribe(ctx_w->ptr);
}

void whisper_print_timings_wrapper(struct whisper_context_wrapper * ctx_w){
    return whisper_print_timings(ctx_w->ptr);
}

void whisper_reset_timings_wrapper(struct whisper_context_wrapper * ctx_w){
    return whisper_reset_timings(ctx_w->ptr);
}

int whisper_encode_wrapper(
        struct whisper_context_wrapper * ctx,
        int   offset,
        int   n_threads){

    return whisper_encode(ctx->ptr, offset, n_threads);
}


int whisper_decode_wrapper(
        struct whisper_context_wrapper * ctx,
        const whisper_token * tokens,
        int   n_tokens,
        int   n_past,
        int   n_threads){
    return whisper_decode(ctx->ptr, tokens, n_tokens, n_past, n_threads);
};

int whisper_tokenize_wrapper(
        struct whisper_context_wrapper * ctx,
        const char * text,
        whisper_token * tokens,
        int   n_max_tokens){
    return whisper_tokenize(ctx->ptr, text, tokens, n_max_tokens);
};

int whisper_lang_auto_detect_wrapper(
    whisper_context_wrapper * ctx,
    int offset_ms,
    int n_threads,
    nb::ndarray<nb::numpy, float, nb::shape<-1>, nb::c_contig> lang_probs) {

    return whisper_lang_auto_detect(
        ctx->ptr, offset_ms, n_threads, lang_probs.data());
}

int whisper_full_wrapper(
    whisper_context_wrapper * ctx_w,
    whisper_full_params params,
    nb::ndarray<nb::numpy, float, nb::shape<-1>, nb::c_contig> samples,
    int n_samples) {

    return whisper_full(ctx_w->ptr, params, samples.data(), n_samples);
}

int whisper_full_parallel_wrapper(
    whisper_context_wrapper * ctx_w,
    whisper_full_params params,
    nb::ndarray<nb::numpy, float, nb::shape<-1>, nb::c_contig> samples,
    int n_samples,
    int n_processors) {

    return whisper_full_parallel(
        ctx_w->ptr, params, samples.data(), n_samples, n_processors);
}

int whisper_full_n_segments_wrapper(struct whisper_context_wrapper * ctx){
    return whisper_full_n_segments(ctx->ptr);
}

int whisper_full_lang_id_wrapper(struct whisper_context_wrapper * ctx){
    return whisper_full_lang_id(ctx->ptr);
}

int64_t whisper_full_get_segment_t0_wrapper(struct whisper_context_wrapper * ctx, int i_segment){
    return whisper_full_get_segment_t0(ctx->ptr, i_segment);
}

int64_t whisper_full_get_segment_t1_wrapper(struct whisper_context_wrapper * ctx, int i_segment){
    return whisper_full_get_segment_t1(ctx->ptr, i_segment);
}

// https://pybind11.readthedocs.io/en/stable/advanced/cast/strings.html
const nb::bytes whisper_full_get_segment_text_wrapper(struct whisper_context_wrapper * ctx, int i_segment){
    const char * c_array = whisper_full_get_segment_text(ctx->ptr, i_segment);
    size_t length = strlen(c_array); // Determine the length of the array
    return nb::bytes(c_array, length); // Return the data without transcoding
};

int whisper_full_n_tokens_wrapper(struct whisper_context_wrapper * ctx, int i_segment){
     return whisper_full_n_tokens(ctx->ptr, i_segment);
}

nb::bytes whisper_full_get_token_text_wrapper(
    whisper_context_wrapper * ctx,
    int i_segment,
    int i_token) {

    const char *str = whisper_full_get_token_text(ctx->ptr, i_segment, i_token);
    return str ? nb::bytes(str, strlen(str)) : nb::bytes();
}

whisper_token whisper_full_get_token_id_wrapper(struct whisper_context_wrapper * ctx, int i_segment, int i_token){
    return whisper_full_get_token_id(ctx->ptr, i_segment, i_token);
}

whisper_token_data whisper_full_get_token_data_wrapper(struct whisper_context_wrapper * ctx, int i_segment, int i_token){
    return whisper_full_get_token_data(ctx->ptr, i_segment, i_token);
}

float whisper_full_get_token_p_wrapper(struct whisper_context_wrapper * ctx, int i_segment, int i_token){
    return whisper_full_get_token_p(ctx->ptr, i_segment, i_token);
}

struct WhisperFullParamsWrapper : public whisper_full_params {
  std::string initial_prompt_str;
  std::string suppress_regex_str;
public:
  nb::object py_progress_callback;
  WhisperFullParamsWrapper(const whisper_full_params& params = whisper_full_params())
    : whisper_full_params(params),
      initial_prompt_str(params.initial_prompt ? params.initial_prompt : ""),
      suppress_regex_str(params.suppress_regex ? params.suppress_regex : "") {
    initial_prompt = initial_prompt_str.empty() ? nullptr : initial_prompt_str.c_str();
    suppress_regex = suppress_regex_str.empty() ? nullptr : suppress_regex_str.c_str();
    // progress callback
    progress_callback_user_data = this;
    progress_callback = [](struct whisper_context* ctx, struct whisper_state* state, int progress, void* user_data) {
      auto* self = static_cast<WhisperFullParamsWrapper*>(user_data);
      if(self && self->print_progress){
        if (self->py_progress_callback) {
          // call the python callback
          nb::gil_scoped_acquire gil;
          self->py_progress_callback(progress);  // Call Python callback
        }
        else {
          fprintf(stderr, "Progress: %3d%%\n", progress);
        } // Default message
      }
    } ;
  }
  WhisperFullParamsWrapper(const WhisperFullParamsWrapper& other)
    : whisper_full_params(static_cast<whisper_full_params>(other)),  // Copy base struct
      initial_prompt_str(other.initial_prompt_str),
      suppress_regex_str(other.suppress_regex_str),
      py_progress_callback(other.py_progress_callback) {
    // Reset pointers to new string copies
    initial_prompt = initial_prompt_str.empty() ? nullptr : initial_prompt_str.c_str();
    suppress_regex = suppress_regex_str.empty() ? nullptr : suppress_regex_str.c_str();
    progress_callback_user_data = this;
    progress_callback = [](struct whisper_context* ctx, struct whisper_state* state, int progress, void* user_data) {
      auto* self = static_cast<WhisperFullParamsWrapper*>(user_data);
      if(self && self->print_progress){
        if (self->py_progress_callback) {
          // call the python callback
          nb::gil_scoped_acquire gil;
          self->py_progress_callback(progress);  // Call Python callback
        }
        else {
          fprintf(stderr, "Progress: %3d%%\n", progress);
        } // Default message
      }
    };
  }
  void set_initial_prompt(const std::string& prompt) {
    initial_prompt_str = prompt;
    initial_prompt = initial_prompt_str.c_str();
  }
  void set_suppress_regex(const std::string& regex) {
    suppress_regex_str = regex;
    suppress_regex = suppress_regex_str.c_str();
  }
};
WhisperFullParamsWrapper  whisper_full_default_params_wrapper(enum whisper_sampling_strategy strategy) {
    return WhisperFullParamsWrapper(whisper_full_default_params(strategy));
}

// callbacks mechanism

void _new_segment_callback(
    whisper_context * ctx,
    whisper_state * state,
    int n_new,
    void * user_data) {

    whisper_context_wrapper ctx_w{ctx};

    nb::gil_scoped_acquire gil;
    if (!py_new_segment_callback.is_none()) {
        py_new_segment_callback(ctx_w, n_new, user_data);
    }
}

void assign_new_segment_callback(
    whisper_full_params * params,
    nb::object f) {

    params->new_segment_callback = _new_segment_callback;
    py_new_segment_callback = std::move(f);
}

bool _encoder_begin_callback(
    whisper_context * ctx,
    whisper_state * state,
    void * user_data) {

    whisper_context_wrapper ctx_w{ctx};

    nb::gil_scoped_acquire gil;
    nb::object result_py = py_encoder_begin_callback(ctx_w, user_data);
    return nb::cast<bool>(result_py);
}

void assign_encoder_begin_callback(struct whisper_full_params *params, nb::object f){
    params->encoder_begin_callback = _encoder_begin_callback;
    py_encoder_begin_callback = f;
}

void _logits_filter_callback(
        struct whisper_context * ctx,
        struct whisper_state * state,
        const whisper_token_data * tokens,
        int   n_tokens,
        float * logits,
        void * user_data){
    struct whisper_context_wrapper ctx_w;
    ctx_w.ptr = ctx;
    // call the python callback
    py_logits_filter_callback(ctx_w, n_tokens, logits, user_data);
}

void assign_logits_filter_callback(struct whisper_full_params *params, nb::object f){
    params->logits_filter_callback = _logits_filter_callback;
    py_logits_filter_callback = f;
}

void whisper_log_callback(enum ggml_log_level level, const char *text, void *user_data) {
    if (py_whisper_log_callback) {
        nb::gil_scoped_acquire gil;
        py_whisper_log_callback(static_cast<int>(level), std::string(text), nb::none());
    }
}

void assign_whisper_log_callback(nb::object f) {
    py_whisper_log_callback = f;
    whisper_log_set(whisper_log_callback, nullptr);  // or replace nullptr if you want user_data
}

nb::dict get_greedy(whisper_full_params * params){
    nb::dict d;
    d["best_of"] = params->greedy.best_of;
    return d;
}

NB_MODULE(_pywhispercpp, m) {
    m.doc() = R"pbdoc(
        Pywhispercpp: Python binding to whisper.cpp
        -----------------------

        .. currentmodule:: _whispercpp

        .. autosummary::
           :toctree: _generate

    )pbdoc";

    m.attr("WHISPER_SAMPLE_RATE") = WHISPER_SAMPLE_RATE;
    m.attr("WHISPER_N_FFT") = WHISPER_N_FFT;
    m.attr("WHISPER_HOP_LENGTH") = WHISPER_HOP_LENGTH;
    m.attr("WHISPER_CHUNK_SIZE") = WHISPER_CHUNK_SIZE;

    nb::class_<whisper_context_wrapper>(m, "whisper_context");
    nb::class_<whisper_token_data>(m,"whisper_token_data")
            .def(nb::init<>())
            .def_rw("id", &whisper_token_data::id)
            .def_rw("tid", &whisper_token_data::tid)
            .def_rw("p", &whisper_token_data::p)
            .def_rw("plog", &whisper_token_data::plog)
            .def_rw("pt", &whisper_token_data::pt)
            .def_rw("ptsum", &whisper_token_data::ptsum)
            .def_rw("t0", &whisper_token_data::t0)
            .def_rw("t1", &whisper_token_data::t1)
            .def_rw("vlen", &whisper_token_data::vlen);

    nb::class_<whisper_model_loader_wrapper>(m,"whisper_model_loader")
            .def(nb::init<>());

    m.def("whisper_init_from_file_with_params", &whisper_init_from_file_wrapper, nb::call_guard<nb::gil_scoped_release>(),
                                                                    "Various functions for loading a ggml whisper model.\n"
                                                                    "Allocate (almost) all memory needed for the model.\n"
                                                                    "Return NULL on failure");

    m.def("whisper_init_from_buffer_with_params", &whisper_init_from_buffer_wrapper, nb::call_guard<nb::gil_scoped_release>(),
                                                                        "Various functions for loading a ggml whisper model.\n"
                                                                        "Allocate (almost) all memory needed for the model.\n"
                                                                        "Return NULL on failure");

    m.def("whisper_init_with_params", &whisper_init_wrapper, nb::call_guard<nb::gil_scoped_release>(),
                                                "Various functions for loading a ggml whisper model.\n"
                                                "Allocate (almost) all memory needed for the model.\n"
                                                "Return NULL on failure");

    m.def("whisper_free", &whisper_free_wrapper, "Frees all memory allocated by the model.");

    m.def("whisper_pcm_to_mel", &whisper_pcm_to_mel_wrapper, nb::call_guard<nb::gil_scoped_release>(),
                                                             "Convert RAW PCM audio to log mel spectrogram.\n"
                                                             "The resulting spectrogram is stored inside the provided whisper context.\n"
                                                             "Returns 0 on success");

    m.def("whisper_set_mel", &whisper_set_mel_wrapper, " This can be used to set a custom log mel spectrogram inside the provided whisper context.\n"
                                                        "Use this instead of whisper_pcm_to_mel() if you want to provide your own log mel spectrogram.\n"
                                                        "n_mel must be 80\n"
                                                        "Returns 0 on success");

    m.def("whisper_encode", &whisper_encode_wrapper, "Run the Whisper encoder on the log mel spectrogram stored inside the provided whisper context.\n"
                                                    "Make sure to call whisper_pcm_to_mel() or whisper_set_mel() first.\n"
                                                    "offset can be used to specify the offset of the first frame in the spectrogram.\n"
                                                    "Returns 0 on success");

    m.def("whisper_decode", &whisper_decode_wrapper, "Run the Whisper decoder to obtain the logits and probabilities for the next token.\n"
                                                    "Make sure to call whisper_encode() first.\n"
                                                    "tokens + n_tokens is the provided context for the decoder.\n"
                                                    "n_past is the number of tokens to use from previous decoder calls.\n"
                                                    "Returns 0 on success\n"
                                                    "TODO: add support for multiple decoders");

    m.def("whisper_tokenize", &whisper_tokenize_wrapper, "Convert the provided text into tokens.\n"
                                                        "The tokens pointer must be large enough to hold the resulting tokens.\n"
                                                        "Returns the number of tokens on success, no more than n_max_tokens\n"
                                                        "Returns -1 on failure\n"
                                                        "TODO: not sure if correct");

    m.def("whisper_lang_max_id", &whisper_lang_max_id, "Largest language id (i.e. number of available languages - 1)");
    m.def("whisper_lang_id", &whisper_lang_id, "Return the id of the specified language, returns -1 if not found\n"
                                                "Examples:\n"
                                                "\"de\" -> 2\n"
                                                "\"german\" -> 2");
    m.def("whisper_lang_str", &whisper_lang_str, "Return the short string of the specified language id (e.g. 2 -> \"de\"), returns nullptr if not found");

    m.def("whisper_lang_auto_detect", &whisper_lang_auto_detect_wrapper, "Use mel data at offset_ms to try and auto-detect the spoken language\n"
                                                                    "Make sure to call whisper_pcm_to_mel() or whisper_set_mel() first\n"
                                                                    "Returns the top language id or negative on failure\n"
                                                                    "If not null, fills the lang_probs array with the probabilities of all languages\n"
                                                                    "The array must be whispe_lang_max_id() + 1 in size\n"
                                                                    "ref: https://github.com/openai/whisper/blob/main/whisper/decoding.py#L18-L69\n");
    m.def("whisper_n_len", &whisper_n_len_wrapper, "whisper_n_len");
    m.def("whisper_n_vocab", &whisper_n_vocab_wrapper, "wrapper_whisper_n_vocab");
    m.def("whisper_n_text_ctx", &whisper_n_text_ctx_wrapper, "whisper_n_text_ctx");
    m.def("whisper_n_audio_ctx", &whisper_n_audio_ctx_wrapper, "whisper_n_audio_ctx");
    m.def("whisper_is_multilingual", &whisper_is_multilingual_wrapper, "whisper_is_multilingual");
    m.def("whisper_get_logits", &whisper_get_logits_wrapper, nb::rv_policy::reference,
                                                            "Token logits obtained from the last call to whisper_decode()\n"
                                                            "The logits for the last token are stored in the last row\n"
                                                            "Rows: n_tokens\n"
                                                            "Cols: n_vocab");

    m.def("whisper_token_to_str", &whisper_token_to_str_wrapper, "whisper_token_to_str");
    m.def("whisper_token_to_bytes", &whisper_token_to_bytes_wrapper, "whisper_token_to_bytes");
    m.def("whisper_token_eot", &whisper_token_eot_wrapper, "whisper_token_eot");
    m.def("whisper_token_sot", &whisper_token_sot_wrapper, "whisper_token_sot");
    m.def("whisper_token_prev", &whisper_token_prev_wrapper);
    m.def("whisper_token_solm", &whisper_token_solm_wrapper);
    m.def("whisper_token_not", &whisper_token_not_wrapper);
    m.def("whisper_token_beg", &whisper_token_beg_wrapper);
    m.def("whisper_token_lang", &whisper_token_lang_wrapper);

    m.def("whisper_token_translate", &whisper_token_translate_wrapper);
    m.def("whisper_token_transcribe", &whisper_token_transcribe_wrapper);

    m.def("whisper_print_timings", &whisper_print_timings_wrapper);
    m.def("whisper_reset_timings", &whisper_reset_timings_wrapper);

    m.def("whisper_print_system_info", &whisper_print_system_info);

    //////////////////////

    nb::enum_<whisper_sampling_strategy>(m, "whisper_sampling_strategy")
        .value("WHISPER_SAMPLING_GREEDY", WHISPER_SAMPLING_GREEDY)
        .value("WHISPER_SAMPLING_BEAM_SEARCH", WHISPER_SAMPLING_BEAM_SEARCH)
        .export_values();

    nb::class_<whisper_full_params>(m, "__whisper_full_params__internal")
        .def(nb::init<>())
        .def("__repr__", [](const whisper_full_params& self) {
            std::ostringstream oss;
            oss << "whisper_full_params("
                << "strategy=" << self.strategy << ", "
                << "n_threads=" << self.n_threads << ", "
                << "n_max_text_ctx=" << self.n_max_text_ctx << ", "
                << "offset_ms=" << self.offset_ms << ", "
                << "duration_ms=" << self.duration_ms << ", "
                << "translate=" << (self.translate ? "True" : "False") << ", "
                << "no_context=" << (self.no_context ? "True" : "False") << ", "
                << "no_timestamps=" << (self.no_timestamps ? "True" : "False") << ", "
                << "single_segment=" << (self.single_segment ? "True" : "False") << ", "
                << "print_special=" << (self.print_special ? "True" : "False") << ", "
                << "print_progress=" << (self.print_progress ? "True" : "False") << ", "
                << "print_realtime=" << (self.print_realtime ? "True" : "False") << ", "
                << "print_timestamps=" << (self.print_timestamps ? "True" : "False") << ", "
                << "token_timestamps=" << (self.token_timestamps ? "True" : "False") << ", "
                << "thold_pt=" << self.thold_pt << ", "
                << "thold_ptsum=" << self.thold_ptsum << ", "
                << "max_len=" << self.max_len << ", "
                << "split_on_word=" << (self.split_on_word ? "True" : "False") << ", "
                << "max_tokens=" << self.max_tokens << ", "
                << "debug_mode=" << (self.debug_mode ? "True" : "False") << ", "
                << "audio_ctx=" << self.audio_ctx << ", "
                << "tdrz_enable=" << (self.tdrz_enable ? "True" : "False") << ", "
                << "suppress_regex=" << (self.suppress_regex ? self.suppress_regex : "None") << ", "
                << "initial_prompt=" << (self.initial_prompt ? self.initial_prompt : "None") << ", "
                << "prompt_tokens=" << (self.prompt_tokens ? "(whisper_token *)" : "None") << ", "
                << "prompt_n_tokens=" << self.prompt_n_tokens << ", "
                << "language=" << (self.language ? self.language : "None") << ", "
                << "detect_language=" << (self.detect_language ? "True" : "False") << ", "
                << "suppress_blank=" << (self.suppress_blank ? "True" : "False") << ", "
                << "suppress_nst=" << (self.suppress_blank ? "True" : "False") << ", "
                << "temperature=" << self.temperature << ", "
                << "max_initial_ts=" << self.max_initial_ts << ", "
                << "length_penalty=" << self.length_penalty << ", "
                << "temperature_inc=" << self.temperature_inc << ", "
                << "entropy_thold=" << self.entropy_thold << ", "
                << "logprob_thold=" << self.logprob_thold << ", "
                << "no_speech_thold=" << self.no_speech_thold << ", "
                << "greedy={best_of=" << self.greedy.best_of << "}, "
                << "beam_search={beam_size=" << self.beam_search.beam_size << ", patience=" << self.beam_search.patience << "}, "
                << "new_segment_callback=" << (self.new_segment_callback ? "(function pointer)" : "None") << ", "
                << "progress_callback=" << (self.progress_callback ? "(function pointer)" : "None") << ", "
                << "encoder_begin_callback=" << (self.encoder_begin_callback ? "(function pointer)" : "None") << ", "
                << "abort_callback=" << (self.abort_callback ? "(function pointer)" : "None") << ", "
                << "logits_filter_callback=" << (self.logits_filter_callback ? "(function pointer)" : "None") << ", "
                << "grammar_rules=" << (self.grammar_rules ? "(whisper_grammar_element **)" : "None") << ", "
                << "n_grammar_rules=" << self.n_grammar_rules << ", "
                << "i_start_rule=" << self.i_start_rule << ", "
                << "grammar_penalty=" << self.grammar_penalty
                << ")";
            return oss.str();
        });

    nb::class_<WhisperFullParamsWrapper, whisper_full_params>(m, "whisper_full_params")
        .def(nb::init<>())
        .def_rw("strategy", &WhisperFullParamsWrapper::strategy)
        .def_rw("n_threads", &WhisperFullParamsWrapper::n_threads)
        .def_rw("n_max_text_ctx", &WhisperFullParamsWrapper::n_max_text_ctx)
        .def_rw("offset_ms", &WhisperFullParamsWrapper::offset_ms)
        .def_rw("duration_ms", &WhisperFullParamsWrapper::duration_ms)
        .def_rw("translate", &WhisperFullParamsWrapper::translate)
        .def_rw("no_context", &WhisperFullParamsWrapper::no_context)
        .def_rw("single_segment", &WhisperFullParamsWrapper::single_segment)
        .def_rw("print_special", &WhisperFullParamsWrapper::print_special)
        .def_rw("print_progress", &WhisperFullParamsWrapper::print_progress)
        .def_rw("progress_callback", &WhisperFullParamsWrapper::py_progress_callback)
        .def_rw("print_realtime", &WhisperFullParamsWrapper::print_realtime)
        .def_rw("print_timestamps", &WhisperFullParamsWrapper::print_timestamps)
        .def_rw("token_timestamps", &WhisperFullParamsWrapper::token_timestamps)
        .def_rw("thold_pt", &WhisperFullParamsWrapper::thold_pt)
        .def_rw("thold_ptsum", &WhisperFullParamsWrapper::thold_ptsum)
        .def_rw("max_len", &WhisperFullParamsWrapper::max_len)
        .def_rw("split_on_word", &WhisperFullParamsWrapper::split_on_word)
        .def_rw("max_tokens", &WhisperFullParamsWrapper::max_tokens)
        .def_rw("audio_ctx", &WhisperFullParamsWrapper::audio_ctx)
        .def_prop_rw("suppress_regex",
            [](WhisperFullParamsWrapper &self) {
                return nb::str(self.suppress_regex ? self.suppress_regex : "");
            },
            [](WhisperFullParamsWrapper &self, const std::string &new_c) {
                self.set_suppress_regex(new_c);
            })
        .def_prop_rw("initial_prompt",
        [](WhisperFullParamsWrapper &self) {
                return nb::str(self.initial_prompt ? self.initial_prompt : "");
            },
            [](WhisperFullParamsWrapper &self, const std::string &initial_prompt) {
                self.set_initial_prompt(initial_prompt);
            }
        )
        .def_rw("prompt_tokens", &WhisperFullParamsWrapper::prompt_tokens)
        .def_rw("prompt_n_tokens", &WhisperFullParamsWrapper::prompt_n_tokens)
        .def_prop_rw("language",
            [](WhisperFullParamsWrapper &self) {
                return nb::str(self.language);
            },
            [](WhisperFullParamsWrapper &self, const char *new_c) {// using lang_id let us avoid issues with memory management
                const int lang_id = (new_c && strlen(new_c) > 0) ? whisper_lang_id(new_c) : -1;
                if (lang_id != -1) {
                    self.language = whisper_lang_str(lang_id);
                } else {
                    self.language = ""; //defaults to auto-detect
                }
            })
        .def_rw("suppress_blank", &WhisperFullParamsWrapper::suppress_blank)
        .def_rw("suppress_nst", &WhisperFullParamsWrapper::suppress_nst)
        .def_rw("temperature", &WhisperFullParamsWrapper::temperature)
        .def_rw("max_initial_ts", &WhisperFullParamsWrapper::max_initial_ts)
        .def_rw("length_penalty", &WhisperFullParamsWrapper::length_penalty)
        .def_rw("temperature_inc", &WhisperFullParamsWrapper::temperature_inc)
        .def_rw("entropy_thold", &WhisperFullParamsWrapper::entropy_thold)
        .def_rw("logprob_thold", &WhisperFullParamsWrapper::logprob_thold)
        .def_rw("no_speech_thold", &WhisperFullParamsWrapper::no_speech_thold)
        // little hack for the internal stuct <undefined type issue>
        .def_prop_rw(
            "greedy",
            [](WhisperFullParamsWrapper &self) {
                nb::dict d;
                d["best_of"] = self.greedy.best_of;
                return d;
            },
            [](WhisperFullParamsWrapper &self, nb::dict d) {
                self.greedy.best_of = nb::cast<int>(d["best_of"]);
            }
        )
        .def_prop_rw(
            "beam_search",
            [](WhisperFullParamsWrapper &self) {
                nb::dict d;
                d["beam_size"] = self.beam_search.beam_size;
                d["patience"]  = self.beam_search.patience;
                return d;
            },
            [](WhisperFullParamsWrapper &self, nb::dict dict) {
                self.beam_search.beam_size = nb::cast<int>(dict["beam_size"]);
                self.beam_search.patience  = nb::cast<float>(dict["patience"]);
            }
        )
        .def_rw("new_segment_callback_user_data", &WhisperFullParamsWrapper::new_segment_callback_user_data)
        .def_rw("encoder_begin_callback_user_data", &WhisperFullParamsWrapper::encoder_begin_callback_user_data)
        .def_rw("logits_filter_callback_user_data", &WhisperFullParamsWrapper::logits_filter_callback_user_data);

    m.def("whisper_full_default_params", &whisper_full_default_params_wrapper);

    m.def("whisper_full", &whisper_full_wrapper, nb::call_guard<nb::gil_scoped_release>(), "Run the entire model: PCM -> log mel spectrogram -> encoder -> decoder -> text\n", "Uses the specified decoding strategy to obtain the text.\n");

    m.def("whisper_full_parallel", &whisper_full_parallel_wrapper, nb::call_guard<nb::gil_scoped_release>(),
                                                                    "Split the input audio in chunks and process each chunk separately using whisper_full()\n"
                                                                    "It seems this approach can offer some speedup in some cases.\n"
                                                                    "However, the transcription accuracy can be worse at the beginning and end of each chunk.");

    m.def("whisper_full_n_segments", &whisper_full_n_segments_wrapper, nb::call_guard<nb::gil_scoped_release>(),
                                                                       "Number of generated text segments.\n"
                                                                       "A segment can be a few words, a sentence, or even a paragraph.\n");

    m.def("whisper_full_lang_id", &whisper_full_lang_id_wrapper, "Language id associated with the current context");
    m.def("whisper_full_get_segment_t0", &whisper_full_get_segment_t0_wrapper, "Get the start time of the specified segment");
    m.def("whisper_full_get_segment_t1", &whisper_full_get_segment_t1_wrapper, "Get the end time of the specified segment");

    m.def("whisper_full_get_segment_text", &whisper_full_get_segment_text_wrapper, "Get the text of the specified segment");
    m.def("whisper_full_n_tokens", &whisper_full_n_tokens_wrapper, "Get number of tokens in the specified segment.");

    m.def("whisper_full_get_token_text", &whisper_full_get_token_text_wrapper, "Get the token text of the specified token in the specified segment.");
    m.def("whisper_full_get_token_id", &whisper_full_get_token_id_wrapper, "Get the token text of the specified token in the specified segment.");

    m.def("whisper_full_get_token_data", &whisper_full_get_token_data_wrapper, "Get token data for the specified token in the specified segment.\n"
                                                                                "This contains probabilities, timestamps, etc.");

    m.def("whisper_full_get_token_p", &whisper_full_get_token_p_wrapper, "Get the probability of the specified token in the specified segment.");

    ////////////////////////////////////////////////////////////////////////////

    m.def("whisper_bench_memcpy", &whisper_bench_memcpy, "Temporary helpers needed for exposing ggml interface");
    m.def("whisper_bench_ggml_mul_mat", &whisper_bench_ggml_mul_mat, "Temporary helpers needed for exposing ggml interface");

    ////////////////////////////////////////////////////////////////////////////
    // Helper mechanism to set callbacks from python
    // The only difference from the C-Style API

    m.def("assign_new_segment_callback", &assign_new_segment_callback, "Assigns a new_segment_callback, takes <whisper_full_params> instance and a callable function with the same parameters which are defined in the interface",
        nb::arg("params"), nb::arg("callback"));

    m.def("assign_encoder_begin_callback", &assign_encoder_begin_callback, "Assigns an encoder_begin_callback, takes <whisper_full_params> instance and a callable function with the same parameters which are defined in the interface",
            nb::arg("params"), nb::arg("callback"));

    m.def("assign_logits_filter_callback", &assign_logits_filter_callback, "Assigns a logits_filter_callback, takes <whisper_full_params> instance and a callable function with the same parameters which are defined in the interface",
            nb::arg("params"), nb::arg("callback"));

    m.def("assign_whisper_log_callback", &assign_whisper_log_callback, "Assign a logging callback for whisper_log_set(level: int, message: str, user_data: None)");

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
